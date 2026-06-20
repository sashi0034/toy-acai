import math

import torch

from ..core import core
from ..simulation_context import SimulationContext


def _collect_alive_fighters(ctx: SimulationContext, team_id: int):
    return [
        fighter
        for fighter in ctx.battlefield.fighters
        if fighter.health > 0.0 and fighter.team_id == team_id
    ]


SELF_FEATURES = 5
ENTITY_FEATURES = 6
OBS_DIM = SELF_FEATURES + 2 * ENTITY_FEATURES + 2 * ENTITY_FEATURES


def build_observation(ctx: SimulationContext):
    values = []
    battlefield_diagonal = ctx.battlefield.battlefield_diagonal_length

    # 自機情報
    fighter = ctx.battlefield.fighters[0]
    values.append(fighter.speed)
    values.append(fighter.missile_cooldown > 0)  # TODO: missile_cooldown_rate

    forward_distance = core.compute_forward_distance_from_boundary(ctx.battlefield, 0)
    values.append(forward_distance.distance / battlefield_diagonal)
    values.append(math.cos(forward_distance.relative_angle))
    values.append(math.sin(forward_distance.relative_angle))

    # 敵機情報
    alive_enemies = _collect_alive_fighters(ctx, team_id=1)
    enemy_futures = []
    for enemy in alive_enemies:
        relative_pose = core.compute_relative_pose(
            core.AbsolutePose(fighter), core.AbsolutePose(enemy)
        )
        enemy_futures.append((relative_pose, enemy.speed))

    # 近い順にソート
    enemy_futures.sort(key=lambda future: future[0].relative_position.length_sq())

    # 最も近い敵機とその次に近い敵機だけ追加
    for i in range(2):
        if i < len(enemy_futures):
            relative_pose, speed = enemy_futures[i]
            values.append(1.0)  # alive
            values.append(relative_pose.relative_position.x / battlefield_diagonal)
            values.append(relative_pose.relative_position.y / battlefield_diagonal)
            values.append(math.cos(relative_pose.relative_yaw))
            values.append(math.sin(relative_pose.relative_yaw))
            values.append(speed)
        else:
            # 敵機がいない場合は 0 で埋める
            values.extend([0.0] * ENTITY_FEATURES)

    # 敵ミサイル情報
    fighter_pose = core.AbsolutePose(fighter)
    missile_futures = []
    for missile in ctx.battlefield.missiles:
        if missile.team_id == fighter.team_id:
            continue
        relative_pose = core.compute_relative_pose(
            fighter_pose, core.AbsolutePose(missile)
        )
        missile_futures.append((relative_pose, missile.speed))

    missile_futures.sort(key=lambda future: future[0].relative_position.length_sq())

    for i in range(2):
        if i < len(missile_futures):
            relative_pose, speed = missile_futures[i]
            values.append(1.0)  # alive
            values.append(relative_pose.relative_position.x / battlefield_diagonal)
            values.append(relative_pose.relative_position.y / battlefield_diagonal)
            values.append(math.cos(relative_pose.relative_yaw))
            values.append(math.sin(relative_pose.relative_yaw))
            values.append(speed)
        else:
            values.extend([0.0] * ENTITY_FEATURES)

    assert len(values) == OBS_DIM, f"Expected {OBS_DIM} values, got {len(values)}"

    return torch.tensor(values, dtype=torch.float32)
