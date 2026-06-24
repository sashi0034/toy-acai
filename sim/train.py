#!/usr/bin/env python3
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

import torch

from . import hyperparameters
from ._slack import create_poster
from .core import core
from .rl.curriculum import CurriculumController
from .rl.observation import OBS_DIM
from .rl.policy_network import PolicyNetwork
from .rl.rollout import (
    Rollout,
    collect_episode,
    initialize_rollout_worker,
    rollout_worker,
    store_state_dict,
)
from .rl.value_network import ValueNetwork
from .simulation_context import SimulationContext, WorkerContext


def _episode_seed(update: int, episode_in_update: int) -> int:
    return update * hyperparameters.EPISODES_PER_UPDATE + episode_in_update


def _render_episode(
    ctx: SimulationContext,
    policy: PolicyNetwork,
    value_network: ValueNetwork,
    curriculum_controller: CurriculumController,
    update: int,
) -> tuple[Path, Rollout]:
    episode_in_update = update % hyperparameters.EPISODES_PER_UPDATE

    # シード値を意図的に巡回させる
    seed = _episode_seed(hyperparameters.NUM_UPDATES, episode_in_update)
    torch.manual_seed(seed)

    render_path = (
        ctx.output_directory() / f"update_{update:04d}_{episode_in_update:04d}.gif"
    )

    # Trails belong to one GIF, so reset the parent-owned renderer for each one.
    ctx.renderer = core.BattlefieldRenderer()
    worker_ctx = WorkerContext(worker_id=-1, seed=seed)
    curriculum = curriculum_controller.create_episode()
    rollout = collect_episode(
        worker_ctx,
        policy,
        value_network,
        curriculum,
        renderer=ctx.renderer,
        render_path=render_path,
    )
    return render_path, rollout


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
        initializer=initialize_rollout_worker,
    ) as executor:
        for update in range(hyperparameters.NUM_UPDATES):
            update_start = time.perf_counter()
            policy_state = store_state_dict(policy_network)
            value_state = store_state_dict(value_network)

            # 複数エピソードを描画なし worker で実行して rollout を収集する
            futures = [
                executor.submit(
                    rollout_worker,
                    episode_in_update,
                    _episode_seed(update, episode_in_update),
                    policy_state,
                    value_state,
                    curriculum_controller,
                )
                for episode_in_update in range(hyperparameters.EPISODES_PER_UPDATE - 1)
            ]

            # ワーカープロセスとは非同期に親プロセスでも実行 (描画あり)
            render_path_result, render_rollout = _render_episode(
                ctx, policy_network, value_network, curriculum_controller, update
            )

            # ワーカーの結果を待機して収集する
            rollouts = [future.result() for future in futures]
            rollouts.append(render_rollout)  # 描画されたエピソードも含める

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
                f"update={update} "
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
