from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from ..core import core
from .. import hyperparameters
from .observation_utils import (
    get_alive_fighters_sorted_by_distance,
    get_missiles_sorted_by_distance,
)


AGENT_FEATURES = 5
OPPONENT_FEATURES = 6
MISSILE_FEATURES = 6
OBS_DIM = AGENT_FEATURES + 2 * OPPONENT_FEATURES + 2 * MISSILE_FEATURES


@dataclass(frozen=True)
class ObservationFeature:
    group: str
    name: str
    value: float


def observation_to_tensor(features: Sequence[ObservationFeature]) -> torch.Tensor:
    return torch.tensor([feature.value for feature in features], dtype=torch.float32)


def build_observation(
    battlefield: core.BattlefieldContext,
) -> list[ObservationFeature]:
    features = []

    def add(group: str, name: str, value: float) -> None:
        features.append(ObservationFeature(group, name, float(value)))

    battlefield_diagonal = battlefield.battlefield_diagonal_length

    distance_factor = 1.0 / (battlefield_diagonal * 0.5)

    # 自機情報
    fighter = battlefield.fighters[0]
    add("AGENT", "speed", fighter.speed * hyperparameters.SPEED_NORMALIZATION_FACTOR)
    add("AGENT", "missile_cooldown", fighter.missile_cooldown > 0)  # TODO: rate

    boundary_distance = core.compute_distance_from_boundary(battlefield, 0)
    add("AGENT", "boundary_distance", boundary_distance.distance * distance_factor)
    add("AGENT", "boundary_angle_cos", math.cos(boundary_distance.relative_angle))
    add("AGENT", "boundary_angle_sin", math.sin(boundary_distance.relative_angle))

    # 敵機情報
    opponent_futures = get_alive_fighters_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    # 最も近い敵機とその次に近い敵機だけ追加
    for i in range(2):
        group = f"OPPONENT[{i}]"
        if i < len(opponent_futures):
            relative_pose, opponent_index = opponent_futures[i]
            opponent = battlefield.fighters[opponent_index]
            add(group, "alive", 1.0)
            add(
                group, "relative_x", relative_pose.relative_position.x * distance_factor
            )
            add(
                group, "relative_y", relative_pose.relative_position.y * distance_factor
            )
            add(
                group,
                "relative_bearing_cos",
                math.cos(relative_pose.relative_bearing),
            )
            add(
                group,
                "relative_bearing_sin",
                math.sin(relative_pose.relative_bearing),
            )
            add(
                group,
                "speed",
                opponent.speed * hyperparameters.SPEED_NORMALIZATION_FACTOR,
            )
        else:
            # 敵機がいない場合は 0 で埋める
            for name in (
                "alive",
                "relative_x",
                "relative_y",
                "relative_bearing_cos",
                "relative_bearing_sin",
                "speed",
            ):
                add(group, name, 0.0)

    # 敵ミサイル情報
    missile_futures = get_missiles_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    for i in range(2):
        group = f"MISSILE[{i}]"
        if i < len(missile_futures):
            relative_pose, missile_index = missile_futures[i]
            missile = battlefield.missiles[missile_index]
            add(group, "alive", 1.0)
            add(
                group, "relative_x", relative_pose.relative_position.x * distance_factor
            )
            add(
                group, "relative_y", relative_pose.relative_position.y * distance_factor
            )
            add(
                group,
                "relative_bearing_cos",
                math.cos(relative_pose.relative_bearing),
            )
            add(
                group,
                "relative_bearing_sin",
                math.sin(relative_pose.relative_bearing),
            )
            add(
                group,
                "speed",
                missile.speed * hyperparameters.SPEED_NORMALIZATION_FACTOR,
            )
        else:
            for name in (
                "alive",
                "relative_x",
                "relative_y",
                "relative_bearing_cos",
                "relative_bearing_sin",
                "speed",
            ):
                add(group, name, 0.0)

    assert len(features) == OBS_DIM, f"Expected {OBS_DIM} features, got {len(features)}"

    return features
