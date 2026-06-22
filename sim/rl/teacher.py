import math
from collections import deque

import torch

from .. import constants
from ..core import core
from .input_utils import copy_inputs as _copy_inputs
from .observation import build_observation, observation_to_tensor
from .curriculum import Curriculum


def try_create_missile_evasion_teacher_data(
    battlefield_history: deque[core.BattlefieldContext],
    inputs_history: deque[list[core.FighterInput]],
    curriculum: Curriculum,
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
        # curriculum.before_step(past_battlefield) # TODO

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


def try_create_boundary_recovery_teacher_data(
    battlefield_history: deque[core.BattlefieldContext],
    inputs_history: deque[list[core.FighterInput]],
    curriculum: Curriculum,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    過去に戻して境界外に出ているとき、境界内に戻るように入力調整
    """

    past_battlefield = core.BattlefieldContext(battlefield_history[0])

    teacher_data = []
    for inputs in inputs_history:
        boundary_distance = core.compute_distance_from_boundary(past_battlefield, 0)
        relative_angle = boundary_distance.relative_angle
        action = core.FighterInput(
            acceleration=math.cos(
                relative_angle
            ),  # 境界と法線方向のときは急減速、逆法線方向のときは急加速
            turn=-1.0 if math.sin(relative_angle) < 0.0 else 1.0,
            fire=False,
        )

        teacher_data.append(
            (
                observation_to_tensor(build_observation(past_battlefield)),
                torch.tensor([action.acceleration, action.turn, float(action.fire)]),
            )
        )

        override_inputs = _copy_inputs(inputs)
        override_inputs[0] = action

        core.update_battlefield(
            past_battlefield, override_inputs, constants.SIMULATION_DELTA_TIME
        )

        if past_battlefield.fighters[0].health <= 0.0:
            # 死亡
            break

    return (
        teacher_data
        if core.compute_distance_from_boundary(past_battlefield, 0).distance >= 0.0
        else []
    )
