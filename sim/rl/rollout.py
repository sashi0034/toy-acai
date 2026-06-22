from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .. import constants, hyperparameters
from ..core import core
from ..simulation_context import WorkerContext, WorkerContextState
from .curriculum import Curriculum, CurriculumController
from .observation import OBS_DIM, build_observation, observation_to_tensor
from .policy_network import PolicyNetwork
from .render_utils import render_observation, render_reward, save_rendered_frames
from .returns import compute_returns, normalize_returns
from .value_network import ValueNetwork
from .input_utils import copy_inputs
from .teacher import (
    try_create_boundary_recovery_teacher_data,
    try_create_missile_evasion_teacher_data,
)


@dataclass
class Rollout:
    """ワーカーが 1 エピソード分収集したデータ"""

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

    context_history: deque[WorkerContextState] = deque(maxlen=30)
    inputs_history: deque[list[core.FighterInput]] = deque(maxlen=30)

    # シミュレーションループ
    with torch.no_grad():
        step = 0
        for step in range(1, max_step_count + 1):
            # 更新直前の状態を直近フレーム分だけ保持する
            context_history.append(ctx.save_state())

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

            # 更新直前の入力を直近フレーム分だけ保持する
            inputs_history.append(copy_inputs(inputs))

            core.update_battlefield(
                battlefield, inputs, constants.SIMULATION_DELTA_TIME
            )

            reward = curriculum.step_reward(
                ctx, context_history[-1].battlefield, inputs
            )

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

    killed_by_missile = any(
        death_event.reason == core.DeathEvent.Reason.HitByMissile
        and death_event.fighter_index == 0
        for death_event in battlefield.death_events
    )
    teacher_data = (
        try_create_missile_evasion_teacher_data(
            context_history, inputs_history, curriculum
        )
        if killed_by_missile
        else try_create_boundary_recovery_teacher_data(
            context_history, inputs_history, curriculum
        )
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


def initialize_rollout_worker() -> None:
    torch.set_num_threads(1)


def rollout_worker(
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


def store_state_dict(model: torch.nn.Module) -> dict[str, np.ndarray]:
    return {
        name: parameter.detach().cpu().numpy().copy()
        for name, parameter in model.state_dict().items()
    }
