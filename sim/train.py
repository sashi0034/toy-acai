#!/usr/bin/env python3
import random
from collections import deque
from pathlib import Path

from PIL import Image
import torch

from .rl.curriculum import Curriculum, initial_curriculum
from .rl.render_utils import render_observation, render_reward, save_rendered_frames
from .rl.observation import OBS_DIM, build_observation
from .rl.policy import Policy
from .rl.returns import compute_returns, normalize_returns

from .core import core
from .simulation_context import SimulationContext
from . import constants
from . import hyperparameters
from ._slack import create_poster


def _copy_inputs(inputs: list[core.FighterInput]) -> list[core.FighterInput]:
    return [
        core.FighterInput(
            fighter_input.acceleration, fighter_input.turn, fighter_input.fire
        )
        for fighter_input in inputs
    ]


def _try_create_teacher_data(
    battlefield_history: deque[core.BattlefieldContext],
    inputs_history: deque[list[core.FighterInput]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    過去に巻き戻し、直前で回避行動可能ならそれを教師データとする
    """

    past_battlefield = core.BattlefieldContext(battlefield_history[0])

    # 過去入力が左旋回をやっていたら、右急旋回を試してみる
    override_turn = (
        1.0
        if sum(inputs_history[i][0].turn for i in range(len(inputs_history))) < 0.0
        else -1.0
    )

    override_action = core.FighterInput(1.0, override_turn, False)
    override_action_tensor = torch.tensor(
        [
            override_action.acceleration,
            override_action.turn,
            float(override_action.fire),
        ]
    )

    teacher_data = []

    # 過去フレームを新入力でシミュレーション
    for inputs in inputs_history:
        # TODO: before_step() が無く、ミサイル発射タイミングが異なる問題への対処

        teacher_data.append(
            (build_observation(past_battlefield), override_action_tensor)
        )

        override_inputs = _copy_inputs(inputs)
        override_inputs[0] = override_action

        core.update_battlefield(
            past_battlefield, override_inputs, constants.SIMULATION_DELTA_TIME
        )

        if past_battlefield.fighters[0].health <= 0.0:
            # 失敗
            return []

    return teacher_data


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
        hyperparameters.MAX_SIMULATION_SECONDS / constants.SIMULATION_DELTA_TIME
    )
    render_every_steps = round(
        constants.RENDER_INTERVAL / constants.SIMULATION_DELTA_TIME
    )

    log_probs = []
    rewards = []
    total_reward = 0.0

    battlefield_history: deque[core.BattlefieldContext] = deque(maxlen=30)
    inputs_history: deque[list[core.FighterInput]] = deque(maxlen=30)

    # シミュレーションループ
    step = 0
    for step in range(1, max_step_count + 1):
        curriculum.before_step(ctx, rng)

        # ニューラルネットワークに入力する観測を構築し、アクションをサンプリングする
        obs_tensor = build_observation(ctx.battlefield).to(device)
        action_tensor, log_prob = policy.sample_action(obs_tensor)
        acceleration, turn, fire = action_tensor.detach().cpu().tolist()

        # 入力
        inputs = [core.FighterInput() for _ in range(core.FIGHTER_COUNT)]

        inputs[0] = core.FighterInput(acceleration, turn, fire >= 0.5)

        for fighter_index, enemy_input in curriculum.opponent_inputs(ctx, rng).items():
            inputs[fighter_index] = enemy_input

        # 更新直前の状態と入力を直近フレーム分だけ保持する
        battlefield_history.append(core.BattlefieldContext(battlefield))
        inputs_history.append(_copy_inputs(inputs))

        core.update_battlefield(battlefield, inputs, constants.SIMULATION_DELTA_TIME)

        reward = curriculum.step_reward(ctx, battlefield_history[-1], inputs)
        rewards.append(reward)
        total_reward += reward
        log_probs.append(log_prob)

        if render:
            renderer.update(battlefield, constants.SIMULATION_DELTA_TIME)

            if step % render_every_steps == 0:
                renderer.render(battlefield)

                frame = Image.fromarray(renderer.image_buffer(), mode="RGBA").copy()
                frame = render_reward(frame, total_reward)
                frame = render_observation(frame, obs_tensor)
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

    hit_by_enemy = any(
        hit_event.target_fighter_index == 0 for hit_event in battlefield.hit_events
    )
    teacher_data = (
        _try_create_teacher_data(battlefield_history, inputs_history)
        if hit_by_enemy
        else []
    )

    return sum(rewards), loss, step, teacher_data


def run():
    ctx = SimulationContext()

    print(f"Output directory: {ctx.output_directory()}")

    poster = create_poster(ctx.output_directory(), Path(__file__).resolve().parents[1])
    poster.start()

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
        teacher_data = []

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

            # 単一エピソードを実行して報酬と損失を計算する
            total_reward, loss, steps, episode_teacher_data = run_episode(
                ctx,
                policy,
                curriculum,
                rng,
                render_path,
            )

            # レンダリング結果投稿
            if render_path is not None and render_path.exists():
                poster.post_file(
                    render_path,
                    f"toy-acai sim update {update + 1}: reward={total_reward:.3f}",
                    f"update_{update:06d}_{episode_in_update:04d}",
                )

            batch_rewards.append(total_reward)
            batch_losses.append(loss)
            batch_steps.append(steps)
            teacher_data.extend(episode_teacher_data)

        # 平均損失を計算し、勾配を更新する
        loss = torch.stack(batch_losses).mean()  # loss = - (1/N) * Σ(log π(a|s) * R)

        # 教師データがある場合は、教師あり学習で追加の勾配更新を行う
        teacher_loss = None
        if teacher_data and (update % hyperparameters.TEACHER_UPDATE_INTERVAL) == 0:
            teacher_observations = torch.stack(
                [observation for observation, _ in teacher_data]
            ).to(device)
            teacher_actions = torch.stack([action for _, action in teacher_data]).to(
                device
            )
            teacher_loss = policy.supervised_loss(teacher_observations, teacher_actions)
            loss += teacher_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"update={update + 1} episodes={hyperparameters.EPISODES_PER_UPDATE} "
            f"curriculum={curriculum.name} "
            f"reward={sum(batch_rewards) / len(batch_rewards):.2f} "
            f"loss={loss.item():.4f} "
            f"teacher_loss={teacher_loss.item() if teacher_loss is not None else 0.0:.4f} "
            f"teacher_samples={len(teacher_data)} "
            f"steps={sum(batch_steps) / len(batch_steps):.1f} "
        )

        next_curriculum = curriculum.after_update(ctx)
        if next_curriculum is not None:
            print(f"Curriculum promoted: {curriculum.name} -> {next_curriculum.name}")
            curriculum = next_curriculum


if __name__ == "__main__":
    run()
