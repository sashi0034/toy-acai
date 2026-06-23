import math
from collections import deque

import torch

from .. import constants
from ..core import core
from ..simulation_context import WorkerContext, WorkerContextState
from .input_utils import copy_inputs as _copy_inputs
from .observation import build_observation, observation_to_tensor
from .curriculum import Curriculum


def try_create_missile_evasion_teacher_data(
    context_history_buffer: deque[WorkerContextState],
    inputs_history_buffer: deque[list[core.FighterInput]],
    curriculum: Curriculum,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    過去に巻き戻し、直前で回避行動可能ならそれを教師データとする
    """

    # 直近 30 フレームの履歴を使う
    context_history = list(context_history_buffer)[-30:]
    inputs_history = list(inputs_history_buffer)[-30:]
    if not context_history:
        return []

    past_ctx = WorkerContext.from_state(context_history[0])

    boundary_distance = core.compute_distance_from_boundary(past_ctx.battlefield, 0)

    override_turn = (  # 境界内
        (  # 過去入力が左旋回をやっていたら、右急旋回を試してみる
            1.0
            if sum(inputs_history[i][0].turn for i in range(len(inputs_history))) < 0.0
            else -1.0
        )
        if boundary_distance.distance > 0.0
        # 境界外にいるなら、境界内に戻る方向に急旋回を試してみる
        else (-1.0 if math.sin(boundary_distance.relative_angle) < 0.0 else 1.0)
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
        curriculum.before_step(past_ctx)

        teacher_data.append(
            (
                observation_to_tensor(build_observation(past_ctx.battlefield)),
                override_action_tensor,
            )
        )

        override_inputs = _copy_inputs(inputs)
        override_inputs[0] = override_action

        core.update_battlefield(
            past_ctx.battlefield, override_inputs, constants.SIMULATION_DELTA_TIME
        )

        if past_ctx.battlefield.fighters[0].health <= 0.0:
            # 失敗
            return []

    return teacher_data


def try_create_boundary_recovery_teacher_data(
    context_history_buffer: deque[WorkerContextState],
    inputs_history_buffer: deque[list[core.FighterInput]],
    curriculum: Curriculum,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    過去に戻して境界外に出ているとき、境界内に戻るように入力調整
    """

    context_history = list(context_history_buffer)
    inputs_history = list(inputs_history_buffer)

    # 逆順から見て境界外に出た最初のフレームを探す
    boundary_entry_index = next(
        (
            index + 1
            for index in range(len(context_history) - 1, -1, -1)
            if context_history[index].battlefield.fighters[0].out_of_bounds_time <= 0.0
        ),
        0,
    )
    if boundary_entry_index == len(context_history):
        return []

    context_history = context_history[boundary_entry_index:]
    inputs_history = inputs_history[boundary_entry_index:]

    past_ctx = WorkerContext.from_state(context_history[0])

    teacher_data = []
    for inputs in inputs_history:
        curriculum.before_step(past_ctx)

        boundary_distance = core.compute_distance_from_boundary(past_ctx.battlefield, 0)
        relative_angle = boundary_distance.relative_angle
        action = core.FighterInput(
            acceleration=-math.cos(
                relative_angle
            ),  # 境界と法線方向のときは急減速、逆法線方向のときは急加速
            turn=-1.0 if math.sin(relative_angle) < 0.0 else 1.0,
            fire=inputs[0].fire,
        )

        teacher_data.append(
            (
                observation_to_tensor(build_observation(past_ctx.battlefield)),
                torch.tensor([action.acceleration, action.turn, float(action.fire)]),
            )
        )

        override_inputs = _copy_inputs(inputs)
        override_inputs[0] = action

        core.update_battlefield(
            past_ctx.battlefield, override_inputs, constants.SIMULATION_DELTA_TIME
        )

        if past_ctx.battlefield.fighters[0].health <= 0.0:
            # 死亡
            break

    return teacher_data
