import math

import torch

from ..core import core
from .observation_utils import (
    get_alive_fighters_sorted_by_distance,
    get_missiles_sorted_by_distance,
)


SELF_FEATURES = 5
ENTITY_FEATURES = 6
OBS_DIM = SELF_FEATURES + 2 * ENTITY_FEATURES + 2 * ENTITY_FEATURES


def build_observation(battlefield: core.BattlefieldContext):
    values = []
    battlefield_diagonal = battlefield.battlefield_diagonal_length

    # 自機情報
    fighter = battlefield.fighters[0]
    values.append(fighter.speed)
    values.append(fighter.missile_cooldown > 0)  # TODO: missile_cooldown_rate

    forward_distance = core.compute_forward_distance_from_boundary(battlefield, 0)
    values.append(forward_distance.distance / battlefield_diagonal)
    values.append(math.cos(forward_distance.relative_angle))
    values.append(math.sin(forward_distance.relative_angle))

    # 敵機情報
    opponent_futures = get_alive_fighters_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    # 最も近い敵機とその次に近い敵機だけ追加
    for i in range(2):
        if i < len(opponent_futures):
            relative_pose, opponent_index = opponent_futures[i]
            opponent = battlefield.fighters[opponent_index]
            values.append(1.0)  # alive
            values.append(relative_pose.relative_position.x / battlefield_diagonal)
            values.append(relative_pose.relative_position.y / battlefield_diagonal)
            values.append(math.cos(relative_pose.relative_yaw))
            values.append(math.sin(relative_pose.relative_yaw))
            values.append(opponent.speed)
        else:
            # 敵機がいない場合は 0 で埋める
            values.extend([0.0] * ENTITY_FEATURES)

    # 敵ミサイル情報
    missile_futures = get_missiles_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    for i in range(2):
        if i < len(missile_futures):
            relative_pose, missile_index = missile_futures[i]
            missile = battlefield.missiles[missile_index]
            values.append(1.0)  # alive
            values.append(relative_pose.relative_position.x / battlefield_diagonal)
            values.append(relative_pose.relative_position.y / battlefield_diagonal)
            values.append(math.cos(relative_pose.relative_yaw))
            values.append(math.sin(relative_pose.relative_yaw))
            values.append(missile.speed)
        else:
            values.extend([0.0] * ENTITY_FEATURES)

    assert len(values) == OBS_DIM, f"Expected {OBS_DIM} values, got {len(values)}"

    return torch.tensor(values, dtype=torch.float32)
