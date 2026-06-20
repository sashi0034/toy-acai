#!/usr/bin/env python3
import random
from pathlib import Path

from PIL import Image
import torch

from .rl.curriculum import Curriculum, initial_curriculum
from .rl.render_utils import render_reward, save_rendered_frames
from .rl.observation import OBS_DIM, build_observation
from .rl.policy import Policy
from .rl.returns import compute_returns, normalize_returns

from .core import core
from .simulation_context import SimulationContext
from . import constants
from . import hyperparameters


def run_episode(
    ctx: SimulationContext,
    policy: Policy,
    curriculum: Curriculum,
    rng: random.Random,
    render_path: Path | None = None,
):
    battlefield = ctx.battlefield
    device = next(policy.parameters()).device
    policy.train()

    render = render_path is not None
    if render:
        ctx.renderer = core.BattlefieldRenderer()
        ctx.renderer.enable_render_to_image_buffer(core.Size(*constants.RENDER_SIZE))
    renderer = ctx.renderer

    curriculum.setup_battlefield(ctx, rng)

    frames = []
    max_step_count = round(
        constants.MAX_SIMULATION_SECONDS / constants.SIMULATION_DELTA_TIME
    )
    render_every_steps = round(
        constants.RENDER_INTERVAL / constants.SIMULATION_DELTA_TIME
    )

    log_probs = []
    rewards = []
    total_reward = 0.0

    # シミュレーションループ
    step = 0
    for step in range(1, max_step_count + 1):
        curriculum.before_step(ctx, rng)

        # ニューラルネットワークに入力する観測を構築し、アクションをサンプリングする
        obs_tensor = build_observation(ctx).to(device)
        action_tensor, log_prob = policy.sample_action(obs_tensor)
        acceleration, turn, fire = action_tensor.detach().cpu().tolist()

        # 入力
        inputs = [core.FighterInput() for _ in range(core.FIGHTER_COUNT)]

        inputs[0] = core.FighterInput(acceleration, turn, fire >= 0.5)

        for fighter_index, enemy_input in curriculum.opponent_inputs(ctx, rng).items():
            inputs[fighter_index] = enemy_input

        previous_battlefield = core.BattlefieldContext(battlefield)

        core.update_battlefield(battlefield, inputs, constants.SIMULATION_DELTA_TIME)

        reward = curriculum.step_reward(ctx, previous_battlefield, inputs)
        rewards.append(reward)
        total_reward += reward
        log_probs.append(log_prob)

        if render:
            renderer.update(battlefield, constants.SIMULATION_DELTA_TIME)

            if step % render_every_steps == 0:
                renderer.render(battlefield)

                frame = Image.fromarray(renderer.image_buffer(), mode="RGBA").copy()
                frame = render_reward(frame, total_reward)
                frames.append(frame)

        if curriculum.is_terminal(ctx):
            break

    # 報酬の割引和を計算し、正規化する
    returns = compute_returns(rewards, hyperparameters.REWARD_DISCOUNT)
    returns = normalize_returns(returns).to(device)

    loss = -(torch.stack(log_probs) * returns).mean()

    if render:
        assert render_path is not None
        save_rendered_frames(frames, render_path, constants.RENDER_INTERVAL)
    curriculum.record_episode()
    return sum(rewards), loss, step


def run():
    ctx = SimulationContext()

    print(f"Output directory: {ctx.output_directory()}")

    # PyTorch 準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = OBS_DIM
    policy = Policy(obs_dim, hidden_dim=hyperparameters.HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparameters.LEARNING_RATE)

    curriculum = initial_curriculum(ctx)

    for update in range(hyperparameters.NUM_UPDATES):
        batch_rewards = []
        batch_losses = []
        batch_steps = []

        # 取り敢えず update ごとに同じ乱数シードを使ってみる
        rng = random.Random(0)

        # 複数エピソードを実行して報酬と損失を収集する
        for episode_in_update in range(hyperparameters.EPISODES_PER_UPDATE):
            render_path = (
                (
                    ctx.output_directory()
                    / f"update_{update:04d}_{episode_in_update:04d}.gif"
                )
                if (episode_in_update == (update % hyperparameters.EPISODES_PER_UPDATE))
                else None
            )

            total_reward, loss, steps = run_episode(
                ctx,
                policy,
                curriculum,
                rng,
                render_path,
            )

            batch_rewards.append(total_reward)
            batch_losses.append(loss)
            batch_steps.append(steps)

        # 平均損失を計算し、勾配を更新する
        loss = torch.stack(batch_losses).mean()  # loss = - (1/N) * Σ(log π(a|s) * R)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"update={update + 1} episodes={hyperparameters.EPISODES_PER_UPDATE} "
            f"curriculum={curriculum.name} "
            f"reward={sum(batch_rewards) / len(batch_rewards):.2f} "
            f"loss={loss.item():.4f} "
            f"steps={sum(batch_steps) / len(batch_steps):.1f} "
        )

        next_curriculum = curriculum.after_update(ctx)
        if next_curriculum is not None:
            print(f"Curriculum promoted: {curriculum.name} -> {next_curriculum.name}")
            curriculum = next_curriculum


if __name__ == "__main__":
    run()
