#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from . import constants, hyperparameters
from ._slack import create_poster
from .core import core
from .rl.curriculum import (
    Curriculum,
    CurriculumController,
)
from .rl.observation import OBS_DIM, build_observation, observation_to_tensor
from .rl.policy_network import PolicyNetwork
from .rl.render_utils import render_observation, render_reward, save_rendered_frames
from .rl.returns import compute_returns, normalize_returns
from .rl.value_network import ValueNetwork
from .simulation_context import SimulationContext, WorkerContext


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
            (
                observation_to_tensor(build_observation(past_battlefield)),
                override_action_tensor,
            )
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


# TODO: ファイル分割


@dataclass
class Rollout:
    """ワーカーが 1 エピソード分収集したデータ。"""

    total_reward: float
    observations: np.ndarray
    raw_actions: np.ndarray
    fires: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    steps: int
    teacher_data: list[tuple[np.ndarray, np.ndarray]]
    is_success: bool


def collect_episode(
    ctx: WorkerContext,
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    curriculum: Curriculum,
    *,
    renderer: core.BattlefieldRenderer | None = None,
    render_path: Path | None = None,
) -> Rollout:
    """Collect one episode without constructing an autograd graph."""
    battlefield = ctx.battlefield
    device = next(policy.parameters()).device

    # 各ワーカーは学習 (train) を行わず、推論 (eval) 専用
    policy.eval()
    value_network.eval()

    render = render_path is not None
    if render:
        assert renderer is not None
        renderer.enable_render_to_image_buffer(core.Size(*constants.RENDER_SIZE))

    curriculum.setup_battlefield(ctx)

    frames: list[Image.Image] = []

    max_step_count = round(
        hyperparameters.MAX_SIMULATION_SECONDS / constants.SIMULATION_DELTA_TIME
    )

    render_every_steps = round(
        constants.RENDER_INTERVAL / constants.SIMULATION_DELTA_TIME
    )

    observations = []
    raw_actions = []
    fires = []
    values = []
    rewards = []
    total_reward = 0.0

    battlefield_history: deque[core.BattlefieldContext] = deque(maxlen=30)
    inputs_history: deque[list[core.FighterInput]] = deque(maxlen=30)

    # シミュレーションループ
    with torch.no_grad():
        step = 0
        for step in range(1, max_step_count + 1):
            curriculum.before_step(ctx)

            # ニューラルネットワークに入力する観測を構築し、アクションをサンプリングする
            observation = build_observation(battlefield)
            obs_tensor = observation_to_tensor(observation).to(device)

            action_tensor, raw_action = policy.sample_action(obs_tensor)
            acceleration, turn, fire = action_tensor.cpu().tolist()

            value = value_network(obs_tensor)

            # 入力
            inputs = [core.FighterInput() for _ in range(core.FIGHTER_COUNT)]

            inputs[0] = core.FighterInput(acceleration, turn, fire >= 0.5)

            for fighter_index, enemy_input in curriculum.opponent_inputs(ctx).items():
                inputs[fighter_index] = enemy_input

            # 更新直前の状態と入力を直近フレーム分だけ保持する
            battlefield_history.append(core.BattlefieldContext(battlefield))
            inputs_history.append(_copy_inputs(inputs))

            core.update_battlefield(
                battlefield, inputs, constants.SIMULATION_DELTA_TIME
            )

            reward = curriculum.step_reward(ctx, battlefield_history[-1], inputs)

            observations.append(obs_tensor.cpu())
            raw_actions.append(raw_action.cpu())
            fires.append(action_tensor[2].cpu())
            values.append(value.cpu())
            rewards.append(reward)

            total_reward += reward

            if render:
                assert renderer is not None
                renderer.update(battlefield, constants.SIMULATION_DELTA_TIME)
                if step % render_every_steps == 0:
                    renderer.render(battlefield)
                    frame = Image.fromarray(renderer.image_buffer(), mode="RGBA").copy()

                    # FIXME: 前フレームの情報が描画されている
                    frame = render_reward(frame, total_reward, value.item())
                    frames.append(render_observation(frame, observation))

            if curriculum.is_terminal(ctx):
                break

    # 報酬計算
    returns = compute_returns(rewards, hyperparameters.REWARD_DISCOUNT)

    # Monte Carlo Advantage
    # FIXME: 分散が大きいので GAE にしたい
    advantages = normalize_returns(returns - torch.stack(values))

    if render:
        assert render_path is not None
        save_rendered_frames(frames, render_path, constants.RENDER_INTERVAL)

    hit_by_enemy = any(
        hit_event.target_fighter_index == 0 for hit_event in battlefield.hit_events
    )
    teacher_data = (
        _try_create_teacher_data(battlefield_history, inputs_history)
        if hit_by_enemy
        else []
    )

    return Rollout(
        total_reward=total_reward,
        observations=torch.stack(observations).numpy(),
        raw_actions=torch.stack(raw_actions).numpy(),
        fires=torch.stack(fires).numpy(),
        returns=returns.numpy(),
        advantages=advantages.numpy(),
        steps=step,
        teacher_data=[
            (observation.numpy(), action.numpy())
            for observation, action in teacher_data
        ],
        is_success=curriculum.is_success(ctx),
    )


def _initialize_rollout_worker() -> None:
    torch.set_num_threads(1)


def _rollout_worker(
    worker_id: int,
    seed: int,
    policy_state: dict[str, np.ndarray],
    value_state: dict[str, np.ndarray],
    curriculum_controller: CurriculumController,
) -> Rollout:
    """Top-level entry point so it can be used with spawn."""
    torch.manual_seed(seed)

    policy = PolicyNetwork(OBS_DIM, hidden_dim=hyperparameters.HIDDEN_DIM)
    policy.load_state_dict(
        {name: torch.from_numpy(parameter) for name, parameter in policy_state.items()}
    )

    value_network = ValueNetwork(OBS_DIM, hidden_dim=hyperparameters.HIDDEN_DIM)
    value_network.load_state_dict(
        {name: torch.from_numpy(parameter) for name, parameter in value_state.items()}
    )

    ctx = WorkerContext(worker_id, seed)
    curriculum = curriculum_controller.create_episode()
    return collect_episode(ctx, policy, value_network, curriculum)


def _store_state_dict(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in model.state_dict().items()
    }


def _episode_seed(update: int, episode_in_update: int) -> int:
    return update * hyperparameters.EPISODES_PER_UPDATE + episode_in_update


def _render_episode(
    ctx: SimulationContext,
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    curriculum_controller: CurriculumController,
    update: int,
) -> Path:
    episode_in_update = update % hyperparameters.EPISODES_PER_UPDATE

    seed = _episode_seed(update, episode_in_update)  # FIXME?
    torch.manual_seed(seed)

    render_path = (
        ctx.output_directory() / f"update_{update:04d}_{episode_in_update:04d}.gif"
    )

    # Trails belong to one GIF, so reset the parent-owned renderer for each one.
    ctx.renderer = core.BattlefieldRenderer()
    worker_ctx = WorkerContext(worker_id=-1, seed=seed)
    curriculum = curriculum_controller.create_episode()
    collect_episode(
        worker_ctx,
        policy,
        value_network,
        curriculum,
        renderer=ctx.renderer,
        render_path=render_path,
    )
    return render_path


def _losses_from_rollout(
    rollout: Rollout,
    policy_network: PolicyNetwork,
    value_network: ValueNetwork,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.from_numpy(rollout.observations).to(device)
    raw_actions = torch.from_numpy(rollout.raw_actions).to(device)
    fires = torch.from_numpy(rollout.fires).to(device)
    advantages = torch.from_numpy(rollout.advantages).to(device)
    returns = torch.from_numpy(rollout.returns).to(device)

    # Actor-Critic の損失は、親プロセスで現在のネットワークを使って再計算する
    log_probs = policy_network.log_prob_from_raw_action(
        observations, raw_actions, fires
    )
    actor_loss = -(log_probs * advantages).mean()
    critic_loss = torch.nn.functional.mse_loss(value_network(observations), returns)
    return actor_loss, critic_loss


def run():
    ctx = SimulationContext()
    print(f"Output directory: {ctx.output_directory()}")
    print(f"Rollout workers: {ctx.rollout_worker_count}")

    poster = create_poster(ctx.output_directory(), Path(__file__).resolve().parents[1])
    poster.start()

    # PyTorch 準備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_network = PolicyNetwork(OBS_DIM, hidden_dim=hyperparameters.HIDDEN_DIM).to(
        device
    )
    policy_optimizer = torch.optim.Adam(
        policy_network.parameters(), lr=hyperparameters.LEARNING_RATE
    )

    value_network = ValueNetwork(OBS_DIM, hidden_dim=hyperparameters.HIDDEN_DIM).to(
        device
    )
    value_optimizer = torch.optim.Adam(
        value_network.parameters(), lr=hyperparameters.LEARNING_RATE
    )

    curriculum_controller = CurriculumController()

    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=ctx.rollout_worker_count,
        mp_context=mp_context,
        initializer=_initialize_rollout_worker,
    ) as executor:
        for update in range(hyperparameters.NUM_UPDATES):
            update_start = time.perf_counter()
            policy_state = _store_state_dict(policy_network)
            value_state = _store_state_dict(value_network)

            # 複数エピソードを描画なし worker で実行して rollout を収集する
            futures = [
                executor.submit(
                    _rollout_worker,
                    episode_in_update,
                    _episode_seed(update, episode_in_update),
                    policy_state,
                    value_state,
                    curriculum_controller,
                )
                for episode_in_update in range(hyperparameters.EPISODES_PER_UPDATE)
            ]

            # ワーカープロセスとは非同期に親プロセスで描画を実行
            # TODO: このプロセスの結果も活用できるはず
            render_path_result = _render_episode(
                ctx, policy_network, value_network, curriculum_controller, update
            )

            # ワーカーの結果を待機して収集する
            rollouts = [future.result() for future in futures]

            policy_network.train()
            value_network.train()

            batch_actor_losses = []
            batch_critic_losses = []
            teacher_data = []
            for rollout in rollouts:
                actor_loss, critic_loss = _losses_from_rollout(
                    rollout, policy_network, value_network, device
                )
                batch_actor_losses.append(actor_loss)
                batch_critic_losses.append(critic_loss)
                teacher_data.extend(rollout.teacher_data)

                curriculum_controller.record_episode(rollout.is_success)

            actor_loss = torch.stack(batch_actor_losses).mean()
            critic_loss = torch.stack(batch_critic_losses).mean()

            # 教師データがある場合は、教師あり学習で追加の勾配更新を行う
            teacher_loss = None
            if teacher_data and (update % hyperparameters.TEACHER_UPDATE_INTERVAL) == 0:
                teacher_observations = torch.stack(
                    [torch.from_numpy(observation) for observation, _ in teacher_data]
                ).to(device)
                teacher_actions = torch.stack(
                    [torch.from_numpy(action) for _, action in teacher_data]
                ).to(device)
                teacher_loss = policy_network.supervised_loss(
                    teacher_observations, teacher_actions
                )

            # 方策更新
            policy_optimizer.zero_grad()
            policy_loss = actor_loss
            if teacher_loss is not None:
                policy_loss += teacher_loss
            policy_loss.backward()
            policy_optimizer.step()

            # 価値関数更新
            value_optimizer.zero_grad()
            critic_loss.backward()
            value_optimizer.step()

            update_elapsed = time.perf_counter() - update_start

            average_reward = sum(rollout.total_reward for rollout in rollouts) / len(
                rollouts
            )

            average_steps = sum(rollout.steps for rollout in rollouts) / len(rollouts)

            message = (
                f"curriculum={curriculum_controller.name} "
                f"update={update + 1} episodes={hyperparameters.EPISODES_PER_UPDATE} "
                f"reward={average_reward:.2f} "
                f"actor_loss={actor_loss.item():.4f} "
                f"critic_loss={critic_loss.item():.4f} "
                f"teacher_loss={teacher_loss.item() if teacher_loss is not None else 'None'} "
                f"teacher_samples={len(teacher_data)} "
                f"steps={average_steps:.1f}\n"
                f"update_elapsed={update_elapsed:.1f}s"
            )
            print(message)
            poster.post_file(
                render_path_result,
                f"{render_path_result.stem}:\n```{message}```",
                render_path_result.stem,
            )

            # カリキュラムの昇格評価
            promotion = curriculum_controller.after_update()
            if promotion is not None:
                previous_name, next_name = promotion
                print(f"Curriculum promoted: {previous_name} -> {next_name}")


if __name__ == "__main__":
    run()
