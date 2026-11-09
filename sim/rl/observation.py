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


AGENT_FEATURES = 2
BOUNDARY_FEATURES = 3
ENTITY_FEATURES = 6

OPPONENT_COUNT = 4
ENEMY_MISSILE_COUNT = 4
AGENT_MISSILE_COUNT = 2

TOKEN_COUNT = 2 + OPPONENT_COUNT + ENEMY_MISSILE_COUNT + AGENT_MISSILE_COUNT

OBS_DIM = (
    AGENT_FEATURES
    + BOUNDARY_FEATURES
    + (OPPONENT_COUNT + ENEMY_MISSILE_COUNT + AGENT_MISSILE_COUNT) * ENTITY_FEATURES
)


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

    # battlefield_diagonal = battlefield.battlefield_diagonal_length
    # distance_factor = 1.0 / (battlefield_diagonal * 0.5)

    # 自機情報
    fighter = battlefield.fighters[0]
    add("AGENT", "speed", fighter.speed * hyperparameters.SPEED_NORMALIZATION_FACTOR)
    add("AGENT", "missile_available", fighter.missile_cooldown > 0)  # TODO: rate

    boundary_distance = core.compute_distance_from_boundary(battlefield, 0)
    add(
        "BOUNDARY",
        "boundary_distance",
        boundary_distance.distance * hyperparameters.DISTANCE_NORMALIZATION_FACTOR,
    )
    add("BOUNDARY", "boundary_angle_cos", math.cos(boundary_distance.relative_angle))
    add("BOUNDARY", "boundary_angle_sin", math.sin(boundary_distance.relative_angle))

    def add_entity(
        group: str,
        relative_pose: core.RelativePose,
        speed: float,
    ) -> None:
        add(group, "alive", 1.0)
        add(
            group,
            "relative_x",
            relative_pose.relative_position.x
            * hyperparameters.DISTANCE_NORMALIZATION_FACTOR,
        )
        add(
            group,
            "relative_y",
            relative_pose.relative_position.y
            * hyperparameters.DISTANCE_NORMALIZATION_FACTOR,
        )
        add(group, "relative_bearing_cos", math.cos(relative_pose.relative_bearing))
        add(group, "relative_bearing_sin", math.sin(relative_pose.relative_bearing))
        add(group, "speed", speed * hyperparameters.SPEED_NORMALIZATION_FACTOR)

    def add_missing_entity(group: str) -> None:
        add(group, "alive", 0.0)
        add(group, "relative_x", 0.0)
        add(group, "relative_y", 0.0)
        add(group, "relative_bearing_cos", 0.0)
        add(group, "relative_bearing_sin", 0.0)
        add(group, "speed", 0.0)

    # 敵機情報
    opponent_futures = get_alive_fighters_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    for i in range(OPPONENT_COUNT):
        group = f"OPPONENT[{i}]"
        if i < len(opponent_futures):
            relative_pose, opponent_index = opponent_futures[i]
            opponent = battlefield.fighters[opponent_index]
            add_entity(group, relative_pose, opponent.speed)
        else:
            add_missing_entity(group)

    # 敵ミサイル情報
    enemy_missile_futures = get_missiles_sorted_by_distance(
        battlefield, core.AbsolutePose(fighter), team_id=1
    )

    for i in range(ENEMY_MISSILE_COUNT):
        group = f"ENEMY_MISSILE[{i}]"
        if i < len(enemy_missile_futures):
            relative_pose, missile_index = enemy_missile_futures[i]
            missile = battlefield.missiles[missile_index]
            add_entity(group, relative_pose, missile.speed)
        else:
            add_missing_entity(group)

    # 自機ミサイル情報
    agent_missile_futures = [
        future
        for future in get_missiles_sorted_by_distance(
            battlefield, core.AbsolutePose(fighter), team_id=fighter.team_id
        )
        # 自軍ミサイルのうち、自機から発射されたミサイルだけを対象にする
        if battlefield.missiles[future[1]].shooter_fighter_index == 0
    ]

    for i in range(AGENT_MISSILE_COUNT):
        group = f"AGENT_MISSILE[{i}]"
        if i < len(agent_missile_futures):
            relative_pose, missile_index = agent_missile_futures[i]
            missile = battlefield.missiles[missile_index]
            add_entity(group, relative_pose, missile.speed)
        else:
            add_missing_entity(group)

    assert len(features) == OBS_DIM, f"Expected {OBS_DIM} features, got {len(features)}"

    return features
