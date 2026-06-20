#!/usr/bin/env python3
import math

from PIL import Image
import torch

from .rl.observation import OBS_DIM, build_observation
from .rl.policy import Policy
from .rl.rule_based_ai import RuleBasedAI
from .rl.returns import compute_returns, normalize_returns

from .core import core
from .simulation_context import SimulationContext, output_path
from . import constants
from . import hyperparameters


def setup_battlefield(ctx: SimulationContext):
    core.init_battlefield(ctx.battlefield)

    for fighter in ctx.battlefield.fighters:
        fighter.health = 0.0

    blue = ctx.battlefield.fighters[0]
    blue.position = core.Vec2(240.0, 450.0)
    blue.yaw = 0.0
    blue.health = 1.0
    blue.missile_cooldown = 0.0
    blue.out_of_bounds_time = 0.0

    for fighter_index, position in ((4, (1360.0, 300.0)), (5, (1360.0, 600.0))):
        fighter = ctx.battlefield.fighters[fighter_index]
        fighter.position = core.Vec2(*position)
        fighter.yaw = math.pi
        fighter.health = 1.0
        fighter.missile_cooldown = 0.0
        fighter.out_of_bounds_time = 0.0


def is_terminal(ctx: SimulationContext) -> bool:
    blue = ctx.battlefield.fighters[0]
    red_alive = any(
        fighter.health > 0.0 and fighter.team_id == 1
        for fighter in ctx.battlefield.fighters
    )
    return blue.health <= 0.0 or not red_alive


def step_reward(ctx: SimulationContext, blue_was_alive: bool) -> float:
    reward = 0.0

    for hit_event in ctx.battlefield.hit_events:
        if hit_event.shooter_fighter_index == 0:
            reward += 1.0

    if blue_was_alive and ctx.battlefield.fighters[0].health <= 0.0:
        reward -= 1.0

    return reward


def save_gif(frames: list[Image.Image], filename: str):
    if not frames:
        print(f"No frames to save: {filename}")
        return

    output_directory = output_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    gif_path = output_directory / filename
    frames[0].save(
        gif_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=round(constants.RENDER_INTERVAL * 1000),
        loop=0,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames to {gif_path}")


def run_episode(
    ctx: SimulationContext,
    policy: Policy,
    render_filename: str | None = None,
):
    battlefield = ctx.battlefield
    device = next(policy.parameters()).device
    policy.train()

    render = render_filename is not None
    if render:
        ctx.renderer = core.BattlefieldRenderer()
        ctx.renderer.enable_render_to_image_buffer(core.Size(*constants.RENDER_SIZE))
    renderer = ctx.renderer

    setup_battlefield(ctx)

    enemy_ai = RuleBasedAI(team_id=1)
    enemy_ai.reset()

    frames = []
    max_step_count = round(
        constants.MAX_SIMULATION_SECONDS / constants.SIMULATION_DELTA_TIME
    )
    render_every_steps = round(
        constants.RENDER_INTERVAL / constants.SIMULATION_DELTA_TIME
    )

    log_probs = []
    rewards = []

    # シミュレーションループ
    step = 0
    for step in range(1, max_step_count + 1):
        obs_tensor = build_observation(ctx).to(device)

        action_tensor, log_prob = policy.sample_action(obs_tensor)
        acceleration, turn, fire = action_tensor.detach().cpu().tolist()

        inputs = [core.FighterInput() for _ in range(core.FIGHTER_COUNT)]
        inputs[0] = core.FighterInput(acceleration, turn, fire >= 0.5)
        for fighter_index, enemy_input in enemy_ai.inputs(ctx).items():
            inputs[fighter_index] = enemy_input

        blue_was_alive = battlefield.fighters[0].health > 0.0

        core.update_battlefield(battlefield, inputs, constants.SIMULATION_DELTA_TIME)
        rewards.append(step_reward(ctx, blue_was_alive))
        log_probs.append(log_prob)

        if render:
            renderer.update(battlefield, constants.SIMULATION_DELTA_TIME)

            if step % render_every_steps == 0:
                renderer.render(battlefield)
                frames.append(
                    Image.fromarray(renderer.image_buffer(), mode="RGBA").copy()
                )

        if is_terminal(ctx):
            break

    # 報酬の割引和を計算し、正規化する
    returns = compute_returns(rewards, hyperparameters.REWARD_DISCOUNT)
    returns = normalize_returns(returns).to(device)

    loss = -(torch.stack(log_probs) * returns).mean()

    if render:
        save_gif(frames, render_filename)
    return sum(rewards), loss, step


def run():
    ctx = SimulationContext()

    # PyTorch 準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = OBS_DIM
    policy = Policy(obs_dim, hidden_dim=hyperparameters.HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hyperparameters.LEARNING_RATE)

    for update in range(hyperparameters.NUM_UPDATES):
        batch_rewards = []
        batch_losses = []
        batch_steps = []

        for episode_in_update in range(hyperparameters.EPISODES_PER_UPDATE):
            render_output = (
                f"update_{update:04d}.gif"
                if episode_in_update == 0 and update % 5 == 0
                else None
            )

            total_reward, loss, steps = run_episode(
                ctx,
                policy,
                render_output,
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
            f"reward={sum(batch_rewards) / len(batch_rewards):.2f} "
            f"loss={loss.item():.4f}"
            f"steps={sum(batch_steps) / len(batch_steps):.1f} "
        )


if __name__ == "__main__":
    run()
