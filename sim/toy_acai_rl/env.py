import itertools
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

TEAM_LEARN = 0
TEAM_RULE = 1

# 観測に入れるミサイル数を固定する。ニューラルネットは入力長が固定である必要があるため、
# 足りない分は 0 で埋める。
MAX_TRACKED_MISSILES = 8
MISSILE_OBS_FEATURES = 7
MAX_SPEED = 360.0
RENDER_INTERVAL = 0.1

SIMULATION_STEPS_PER_SECOND = 60
RECOVERY_ROLLBACK_STEPS = int(1.5 * SIMULATION_STEPS_PER_SECOND)
RECOVERY_SEGMENT_STEPS = int(0.5 * SIMULATION_STEPS_PER_SECOND)
RECOVERY_SEGMENT_COUNT = 4
RECOVERY_ROLLOUT_STEPS = RECOVERY_SEGMENT_STEPS * RECOVERY_SEGMENT_COUNT
RECOVERY_TEACHER_STEPS = RECOVERY_SEGMENT_STEPS
# action は [acceleration, turn, fire]。左旋回は yaw を減らす向きとして扱う。
RECOVERY_EXTREME_ACTIONS = (
    np.array([1.0, -1.0, 1.0], dtype=np.float64),
    np.array([-1.0, -1.0, 1.0], dtype=np.float64),
    np.array([1.0, 1.0, 1.0], dtype=np.float64),
    np.array([-1.0, 1.0, 1.0], dtype=np.float64),
)

MISSILE_COLUMN_COUNT = 9
MISSILE_TEAM_COLUMN = 6

SEEKER_HALF_ANGLE = 0.85

# -----------------------------------------------

RANDOM_START_X_RANGE = (0.08, 0.92)
RANDOM_START_Y_RANGE = (0.12, 0.88)
MIN_TEAM_START_DISTANCE = 240.0
RANDOM_START_MAX_ATTEMPTS = 4096

RANDOM_START_YAW_JITTER = 0.45
LOW_MOVEMENT_WINDOW_STEPS = SIMULATION_STEPS_PER_SECOND

AUX_KILL_REWARD = 5.0
AUX_DEATH_PENALTY = 5.0
# AUX_SURVIVAL_REWARD_PER_STEP = 0.0015 # 生存報酬は一旦無効化する。必要になったら再度有効化する。
AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP = 0.01
AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP = 0.05
AUX_MISSILE_TRACKING_PENALTY_DISTANCE_SCALE = 200.0
AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP = 0.01
AUX_NEAREST_ENEMY_FACING_ME_PENALTY_PER_STEP = 0.015
AUX_LOW_MOVEMENT_PENALTY_PER_STEP = 0.02
AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD = 250.0
AUX_MISSILE_FIRE_REWARD = 1.0

# -----------------------------------------------


def add_default_module_paths(
    repo_root: Path, module_dir: Optional[Path] = None
) -> None:
    # C++/Python バインディングである toy_acai_core を import できるように、
    # ビルド済みモジュールの候補ディレクトリを sys.path に追加する。
    if module_dir is not None:
        sys.path.insert(0, str(module_dir.resolve()))
    for path in (repo_root / "linux-python" / "build", repo_root / "build"):
        if path.exists():
            sys.path.insert(0, str(path))


def load_core(repo_root: Path, module_dir: Optional[Path] = None):
    add_default_module_paths(repo_root, module_dir)
    import toy_acai_core

    return toy_acai_core


def _angle_delta(target: float, current: float) -> float:
    # 角度差を [-pi, pi] に丸める。旋回方向を決めるときに扱いやすい形。
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


def _alive(fighters: np.ndarray) -> np.ndarray:
    return fighters[:, 6] > 0.0


def _copy_observation(obs: Dict[str, object]) -> Dict[str, object]:
    copied = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _min_pairwise_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    diff = points_a[:, None, :] - points_b[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


def _team_indices(
    fighters: np.ndarray,
    team_id: int,
    active_count: Optional[int] = None,
) -> np.ndarray:
    indices = np.where(fighters[:, 0] == team_id)[0]
    if active_count is None:
        return indices
    return indices[: max(0, int(active_count))]


class RuleBasedOpponent:
    """Simple red-team controller: point at nearest living blue fighter and fire."""

    def __init__(self, team_id: int = TEAM_RULE):
        self.team_id = team_id

    def actions(self, obs: Dict[str, np.ndarray], fighter_count: int) -> np.ndarray:
        fighters = np.asarray(obs["fighters"], dtype=np.float64)
        actions = np.zeros((fighter_count, 3), dtype=np.float64)
        field_w = float(obs["battlefield"][2])
        field_h = float(obs["battlefield"][3])
        alive = _alive(fighters)
        target_team = TEAM_RULE if self.team_id == TEAM_LEARN else TEAM_LEARN
        target_indices = np.where((fighters[:, 0] == target_team) & alive)[0]

        for i, fighter in enumerate(fighters):
            if int(fighter[0]) != self.team_id or fighter[6] <= 0.0:
                continue

            actions[i, 0] = 0.55
            if len(target_indices) == 0:
                actions[i, 1] = _edge_turn(fighter, field_w, field_h)
                continue

            deltas = fighters[target_indices, 2:4] - fighter[2:4]
            distances = np.sum(deltas * deltas, axis=1)
            target_delta = deltas[int(np.argmin(distances))]
            target_yaw = math.atan2(float(target_delta[1]), float(target_delta[0]))
            yaw_delta = _angle_delta(target_yaw, float(fighter[4]))
            actions[i, 1] = np.clip(yaw_delta / 0.7, -1.0, 1.0)
            actions[i, 2] = 1.0 if abs(yaw_delta) < 0.35 else 0.0

        return actions


def _edge_turn(fighter: np.ndarray, field_w: float, field_h: float) -> float:
    x = float(fighter[2])
    y = float(fighter[3])
    if 80.0 <= x <= field_w - 80.0 and 80.0 <= y <= field_h - 80.0:
        return 0.0
    center_yaw = math.atan2(field_h * 0.5 - y, field_w * 0.5 - x)
    return float(np.clip(_angle_delta(center_yaw, float(fighter[4])) / 0.7, -1.0, 1.0))


def _ray_to_boundary_distance(
    x: float,
    y: float,
    ray_dx: float,
    ray_dy: float,
    field_w: float,
    field_h: float,
    diag: float,
) -> float:
    # 自機から指定方向へ線を伸ばし、戦場の端までの距離を 0..1 に正規化して返す。
    # 端が近い方向を観測に入れることで、エリア外へ出る前に曲がる手がかりになる。
    candidates = []
    eps = 1e-9
    if abs(ray_dx) > eps:
        candidates.extend([(0.0 - x) / ray_dx, (field_w - x) / ray_dx])
    if abs(ray_dy) > eps:
        candidates.extend([(0.0 - y) / ray_dy, (field_h - y) / ray_dy])

    for t in sorted(candidate for candidate in candidates if candidate >= 0.0):
        hit_x = x + ray_dx * t
        hit_y = y + ray_dy * t
        if -eps <= hit_x <= field_w + eps and -eps <= hit_y <= field_h + eps:
            return float(np.clip(t / diag, 0.0, 1.0))
    return 0.0


def _boundary_ray_features(
    x: float,
    y: float,
    forward: np.ndarray,
    right: np.ndarray,
    field_w: float,
    field_h: float,
    diag: float,
) -> list:
    # 前・左・右の 3 方向について、境界までの距離を観測特徴量にする。
    directions = (
        forward,
        -right,
        right,
    )
    features = []
    for direction in directions:
        norm = max(float(np.linalg.norm(direction)), 1e-9)
        ray = direction / norm
        features.append(
            _ray_to_boundary_distance(
                x,
                y,
                float(ray[0]),
                float(ray[1]),
                field_w,
                field_h,
                diag,
            )
        )
    return features


def build_agent_observations(
    obs: Dict[str, np.ndarray],
    learner_team: int = TEAM_LEARN,
    active_learner_count: Optional[int] = None,
) -> np.ndarray:
    # シミュレータの生の状態(dict)を、ニューラルネットへ入れられる固定長ベクトルへ変換する。
    # 座標は自機基準の forward/right 成分に分け、マップサイズで割ってスケールをそろえる。
    fighters = np.asarray(obs["fighters"], dtype=np.float64)
    missiles = np.asarray(obs["missiles"], dtype=np.float64)
    field_w = float(obs["battlefield"][2])
    field_h = float(obs["battlefield"][3])
    diag = max(math.hypot(field_w, field_h), 1e-6)

    agent_indices = _team_indices(fighters, learner_team, active_learner_count)
    all_obs = []
    for agent_idx in agent_indices:
        # ここから 1 機ぶんの観測を作る。
        fighter = fighters[agent_idx]
        features = []
        x = float(fighter[2])
        y = float(fighter[3])
        yaw = float(fighter[4])
        speed = float(fighter[5])
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        velocity = forward * speed
        boundary_rays = _boundary_ray_features(
            x,
            y,
            forward,
            right,
            field_w,
            field_h,
            diag,
        )
        features.extend(
            boundary_rays
            + [
                # 自機の基本状態。速度と生存フラグを観測に入れる。
                speed / MAX_SPEED,
                1.0 if float(fighter[6]) > 0.0 else 0.0,
            ]
        )

        others = [i for i in range(len(fighters)) if i != agent_idx]

        def other_sort_key(i):
            other = fighters[i]
            rel = other[2:4] - fighter[2:4]
            distance = math.hypot(float(rel[0]), float(rel[1]))
            is_friendly = int(other[0]) == learner_team
            is_dead = float(other[6]) <= 0.0
            return (is_friendly, is_dead, distance, i)

        # 敵を先、味方を後にし、それぞれ生存機を距離順、撃墜済み機体を後ろに並べる。
        others.sort(key=other_sort_key)
        for other_idx in others:
            other = fighters[other_idx]
            rel = other[2:4] - fighter[2:4]
            other_yaw = float(other[4])
            other_speed = float(other[5])
            distance = math.hypot(float(rel[0]), float(rel[1]))
            bearing = math.atan2(float(rel[1]), float(rel[0]))
            bearing_delta = _angle_delta(bearing, yaw)
            other_forward = np.array(
                [math.cos(other_yaw), math.sin(other_yaw)], dtype=np.float64
            )
            rel_velocity = other_forward * other_speed - velocity
            closing = 0.0
            if distance > 1e-6:
                closing = -float(np.dot(rel, rel_velocity)) / (distance * MAX_SPEED)
            in_fire_arc = 1.0 if abs(bearing_delta) <= SEEKER_HALF_ANGLE else 0.0
            # 他機との関係は「自機から見て前後どちらか・左右どちらか」を中心に表す。
            # 絶対座標よりも、旋回や射撃の判断に直接つながりやすい。
            features.extend(
                [
                    distance / diag,
                    math.cos(bearing_delta),
                    math.sin(bearing_delta),
                    math.cos(_angle_delta(other_yaw, yaw)),
                    math.sin(_angle_delta(other_yaw, yaw)),
                    other_speed / MAX_SPEED,
                    1.0 if float(other[6]) > 0.0 else 0.0,
                    1.0 if int(other[0]) == learner_team else -1.0,
                    float(np.clip(closing, -2.0, 2.0)),
                    in_fire_arc,
                    1.0 if float(other[7]) <= 0.0 else 0.0,
                ]
            )

        missile_features = []
        for missile in missiles:
            if int(missile[MISSILE_TEAM_COLUMN]) == learner_team:
                continue
            rel = missile[0:2] - fighter[2:4]
            distance = math.hypot(float(rel[0]), float(rel[1]))
            missile_features.append((distance, missile, rel))
        missile_features.sort(key=lambda item: item[0])
        # 近いミサイルほど回避に重要なので、距離順に最大 MAX_TRACKED_MISSILES 個だけ見る。
        for _, missile, rel in missile_features[:MAX_TRACKED_MISSILES]:
            distance = math.hypot(float(rel[0]), float(rel[1]))
            bearing = math.atan2(float(rel[1]), float(rel[0]))
            bearing_delta = _angle_delta(bearing, yaw)
            missile_yaw = float(missile[2])
            missile_forward = np.array(
                [math.cos(missile_yaw), math.sin(missile_yaw)], dtype=np.float64
            )
            missile_closing = 0.0
            incoming_alignment = 0.0
            if distance > 1e-6:
                missile_velocity = missile_forward * float(missile[3])
                missile_closing = -float(np.dot(rel, missile_velocity - velocity)) / (
                    distance * MAX_SPEED
                )
                incoming_alignment = -float(np.dot(rel / distance, missile_forward))
            # missile_closing や incoming_alignment は「自分へ向かって来ているか」の手がかり。
            # 単に近いだけでなく、危険度を学習しやすくするために入れている。
            features.extend(
                [
                    distance / diag,
                    float(np.clip(missile_closing, -2.0, 2.0)),
                    math.cos(bearing_delta),
                    math.sin(bearing_delta),
                    math.cos(_angle_delta(missile_yaw, yaw)),
                    math.sin(_angle_delta(missile_yaw, yaw)),
                    float(np.clip(incoming_alignment, -1.0, 1.0)),
                ]
            )
        for _ in range(
            MAX_TRACKED_MISSILES - len(missile_features[:MAX_TRACKED_MISSILES])
        ):
            # 入力長を固定するため、見えているミサイルが少ない場合は 0 埋めする。
            features.extend([0.0] * MISSILE_OBS_FEATURES)

        all_obs.append(features)

    return np.asarray(all_obs, dtype=np.float32)


def observation_dim(toy_acai_core) -> int:
    env = toy_acai_core.BattlefieldEnv()
    obs = env.reset()
    return int(build_agent_observations(obs).shape[1])


def terminal_score(
    *,
    blue_alive: int,
    red_alive: int,
    episode_steps: int,
    max_steps: int,
    team_size: int,
    opponent_team_size: Optional[int] = None,
) -> float:
    # エピソード終了時の大きな報酬。
    # 勝敗を強く教え、勝った場合は味方生存数と早さを少しだけ加点する。
    team_size = max(1, int(team_size))
    opponent_team_size = max(1, int(opponent_team_size or team_size))
    blue_alive_ratio = float(blue_alive) / team_size
    red_alive_ratio = float(red_alive) / opponent_team_size
    if red_alive == 0:
        time_bonus = 0.0
        if max_steps > 0:
            time_bonus = np.clip((max_steps - episode_steps) / max_steps, 0.0, 1.0)
        return float(2.0 + 0.5 * blue_alive_ratio + 0.05 * time_bonus)
    return float(-2.0 * red_alive_ratio + 0.2 * blue_alive_ratio)


def _is_facing_position(
    fighter: np.ndarray,
    target_position: np.ndarray,
    half_angle: float = SEEKER_HALF_ANGLE,
) -> bool:
    rel = target_position - fighter[2:4]
    distance = math.hypot(float(rel[0]), float(rel[1]))
    if distance <= 1e-6:
        return False
    bearing = math.atan2(float(rel[1]), float(rel[0]))
    return abs(_angle_delta(bearing, float(fighter[4]))) <= half_angle


def _nearest_living_opponent_index(
    fighter: np.ndarray,
    fighters: np.ndarray,
    opponent_team: int,
) -> Optional[int]:
    opponent_indices = np.where(
        (fighters[:, 0] == float(opponent_team)) & (fighters[:, 6] > 0.0)
    )[0]
    nearest_idx = None
    nearest_distance_sq = math.inf
    for opponent_idx in opponent_indices:
        rel = fighters[opponent_idx, 2:4] - fighter[2:4]
        distance_sq = float(np.dot(rel, rel))
        if distance_sq <= 1e-12 or nearest_distance_sq <= distance_sq:
            continue
        nearest_distance_sq = distance_sq
        nearest_idx = int(opponent_idx)
    return nearest_idx


def _missile_tracking_penalty(
    missile: np.ndarray,
    fighter: np.ndarray,
    *,
    max_penalty: float,
    distance_scale: float,
) -> float:
    rel = fighter[2:4] - missile[0:2]
    distance = math.hypot(float(rel[0]), float(rel[1]))
    if distance <= 1e-6:
        return float(max_penalty)
    target_yaw = math.atan2(float(rel[1]), float(rel[0]))
    if abs(_angle_delta(target_yaw, float(missile[2]))) > SEEKER_HALF_ANGLE:
        return 0.0
    raw_penalty = (
        float(max_penalty)
        * float(distance_scale)
        / max(
            distance,
            float(distance_scale),
        )
    )
    return float(np.clip(raw_penalty, 0.0, float(max_penalty)))


def _has_living_opponent_in_fire_arc(
    fighter: np.ndarray,
    fighters: np.ndarray,
    opponent_team: int,
) -> bool:
    # C++ 側のミサイルロック条件と同じく、前方の射界内に生存敵がいれば True。
    shooter_position = fighter[2:4]
    shooter_yaw = float(fighter[4])
    opponent_indices = np.where(
        (fighters[:, 0] == float(opponent_team)) & (fighters[:, 6] > 0.0)
    )[0]
    for opponent_idx in opponent_indices:
        rel = fighters[opponent_idx, 2:4] - shooter_position
        distance = math.hypot(float(rel[0]), float(rel[1]))
        if distance <= 1e-6:
            continue
        bearing = math.atan2(float(rel[1]), float(rel[0]))
        if abs(_angle_delta(bearing, shooter_yaw)) <= SEEKER_HALF_ANGLE:
            return True
    return False


def auxiliary_agent_rewards(
    obs: Dict[str, np.ndarray],
    previous_obs: Optional[Dict[str, np.ndarray]] = None,
    learner_team: int = TEAM_LEARN,
    opponent_team: int = TEAM_RULE,
    kill_reward: float = AUX_KILL_REWARD,
    death_penalty: float = AUX_DEATH_PENALTY,
    out_of_bounds_penalty_per_step: float = AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP,
    missile_tracking_penalty_max_per_step: float = (
        AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP
    ),
    missile_tracking_penalty_distance_scale: float = (
        AUX_MISSILE_TRACKING_PENALTY_DISTANCE_SCALE
    ),
    low_movement_penalty_per_step: float = AUX_LOW_MOVEMENT_PENALTY_PER_STEP,
    low_movement_distance_threshold: float = AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD,
    missile_fire_reward: float = AUX_MISSILE_FIRE_REWARD,
    low_movement_distances_1s: Optional[np.ndarray] = None,
    learner_count: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # 毎ステップ与える補助報酬。終端報酬だけだと「何が良かったか」が遠すぎるため、
    # 場外・角度関係・撃墜・損失を小さな手がかりとして追加して学習を助ける。
    fighters = np.asarray(obs["fighters"], dtype=np.float64)
    missiles = np.asarray(obs["missiles"], dtype=np.float64)
    hit_events = np.asarray(
        obs.get("hit_events", np.zeros((0, 4), dtype=np.float64)),
        dtype=np.float64,
    )
    if hit_events.ndim == 1:
        hit_events = hit_events.reshape((-1, 4))

    learner_indices = _team_indices(fighters, learner_team, learner_count)
    rewards = np.zeros((len(learner_indices),), dtype=np.float32)

    learner_fighters = fighters[learner_indices]
    alive = learner_fighters[:, 6] > 0.0
    in_bounds = learner_fighters[:, 8] <= 0.0
    out_of_bounds = alive & ~in_bounds
    survival_reward = 0.0
    # 生存報酬は実験のため一旦無効化している。
    # survival_eligible = alive & in_bounds
    # rewards[survival_eligible] += float(survival_reward_per_step)

    out_of_bounds_penalties = out_of_bounds.astype(np.float64) * float(
        out_of_bounds_penalty_per_step
    )
    rewards -= out_of_bounds_penalties.astype(np.float32)
    out_of_bounds_penalty_total = float(np.sum(out_of_bounds_penalties))

    if missiles.ndim == 1:
        missiles = missiles.reshape((-1, MISSILE_COLUMN_COUNT))

    missile_tracking_penalty_total = 0.0
    nearest_enemy_facing_reward_total = 0.0
    nearest_enemy_facing_penalty_total = 0.0
    low_movement_penalty_total = 0.0
    mean_movement_distance_1s = 0.0
    missile_fire_reward_total = 0.0
    opponent_indices = np.where(
        (fighters[:, 0] == float(opponent_team)) & (fighters[:, 6] > 0.0)
    )[0]

    for agent_idx, fighter_idx in enumerate(learner_indices):
        if not alive[agent_idx]:
            continue
        fighter = fighters[int(fighter_idx)]
        for missile in missiles:
            if int(missile[MISSILE_TEAM_COLUMN]) == learner_team:
                continue
            penalty = _missile_tracking_penalty(
                missile,
                fighter,
                max_penalty=missile_tracking_penalty_max_per_step,
                distance_scale=missile_tracking_penalty_distance_scale,
            )
            if penalty <= 0.0:
                continue
            rewards[agent_idx] -= np.float32(penalty)
            missile_tracking_penalty_total += penalty

        if opponent_indices.shape[0] == 0:
            continue
        nearest_opponent_idx = _nearest_living_opponent_index(
            fighter,
            fighters,
            opponent_team,
        )
        if nearest_opponent_idx is None:
            continue
        nearest_opponent = fighters[nearest_opponent_idx]
        if _is_facing_position(fighter, nearest_opponent[2:4]):
            rewards[agent_idx] += np.float32(AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP)
            nearest_enemy_facing_reward_total += (
                AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP
            )
        if _is_facing_position(nearest_opponent, fighter[2:4]):
            rewards[agent_idx] -= np.float32(
                AUX_NEAREST_ENEMY_FACING_ME_PENALTY_PER_STEP
            )
            nearest_enemy_facing_penalty_total += (
                AUX_NEAREST_ENEMY_FACING_ME_PENALTY_PER_STEP
            )

    if low_movement_distances_1s is not None:
        movement_distances_1s = np.asarray(
            low_movement_distances_1s,
            dtype=np.float64,
        ).reshape((-1,))
        if movement_distances_1s.shape[0] < len(learner_indices):
            padded = np.full((len(learner_indices),), np.nan, dtype=np.float64)
            padded[: movement_distances_1s.shape[0]] = movement_distances_1s
            movement_distances_1s = padded
        else:
            movement_distances_1s = movement_distances_1s[: len(learner_indices)]
        valid_movement = np.isfinite(movement_distances_1s)
        movement_eligible = alive & in_bounds & valid_movement
        low_movement = movement_eligible & (
            movement_distances_1s <= float(low_movement_distance_threshold)
        )
        low_movement_penalties = low_movement.astype(np.float64) * float(
            low_movement_penalty_per_step
        )
        rewards -= low_movement_penalties.astype(np.float32)
        low_movement_penalty_total = float(np.sum(low_movement_penalties))
        if np.any(movement_eligible):
            mean_movement_distance_1s = float(
                np.mean(movement_distances_1s[movement_eligible])
            )

    if previous_obs is not None:
        previous_fighters = np.asarray(previous_obs["fighters"], dtype=np.float64)
        previous_alive = previous_fighters[learner_indices, 6] > 0.0
        previous_cooldown = previous_fighters[learner_indices, 7]
        current_cooldown = learner_fighters[:, 7]
        aimed_at_opponent = np.array(
            [
                _has_living_opponent_in_fire_arc(
                    previous_fighters[int(fighter_idx)],
                    previous_fighters,
                    opponent_team,
                )
                for fighter_idx in learner_indices
            ],
            dtype=bool,
        )
        launched = (
            alive
            & previous_alive
            & (current_cooldown > previous_cooldown + 1e-6)
            & aimed_at_opponent
        )
        fire_rewards = launched.astype(np.float64) * float(missile_fire_reward)
        rewards += fire_rewards.astype(np.float32)
        missile_fire_reward_total = float(np.sum(fire_rewards))

    fighter_to_agent = {
        int(fighter_idx): agent_idx
        for agent_idx, fighter_idx in enumerate(learner_indices)
    }
    blue_kills = 0
    for hit_event in hit_events:
        shooter_idx = int(hit_event[0])
        shooter_team = int(hit_event[1])
        target_team = int(hit_event[3])
        agent_idx = fighter_to_agent.get(shooter_idx)
        if (
            agent_idx is None
            or shooter_team != learner_team
            or target_team != opponent_team
        ):
            continue
        # 撃墜した本人には大きめの報酬を与える。
        rewards[agent_idx] += float(kill_reward)
        blue_kills += 1
    kill_reward_total = float(blue_kills * kill_reward)
    blue_losses = 0
    if previous_obs is not None:
        for fighter_idx in learner_indices:
            fighter_idx = int(fighter_idx)
            if (
                previous_fighters[fighter_idx, 6] <= 0.0
                or fighters[fighter_idx, 6] > 0.0
            ):
                continue
            agent_idx = fighter_to_agent[fighter_idx]
            # 前ステップでは生存、今ステップでは非生存なら、その機が撃墜されたとみなす。
            rewards[agent_idx] -= float(death_penalty)
            blue_losses += 1

    death_penalty_total = float(blue_losses * death_penalty)
    info = {
        "survival_reward": survival_reward,
        "out_of_bounds_penalty": out_of_bounds_penalty_total,
        "missile_tracking_penalty": missile_tracking_penalty_total,
        "nearest_enemy_facing_reward": nearest_enemy_facing_reward_total,
        "nearest_enemy_facing_penalty": nearest_enemy_facing_penalty_total,
        "low_movement_penalty": low_movement_penalty_total,
        "mean_movement_distance_1s": mean_movement_distance_1s,
        "missile_fire_reward": missile_fire_reward_total,
        "kill_reward": kill_reward_total,
        "death_penalty": death_penalty_total,
        "blue_kills": float(blue_kills),
        "blue_losses": float(blue_losses),
        "hit_events": float(hit_events.shape[0]),
    }
    return rewards, info


class StepResult:
    """env.step の返り値を分かりやすくまとめる小さな入れ物。"""

    def __init__(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        done: bool,
        info: Dict[str, float],
    ):
        self.observations = observations
        self.rewards = rewards
        self.done = done
        self.info = info


class RecoveryTeacherExample:
    def __init__(self, agent_id: int, observation: np.ndarray, action: np.ndarray):
        self.agent_id = agent_id
        self.observation = observation
        self.action = action


class _RecoveryHistoryEntry:
    def __init__(self, step_count: int, snapshot: object, obs: Dict[str, object]):
        self.step_count = step_count
        self.snapshot = snapshot
        self.obs = obs


class ToyAcaiPPOEnv:
    """toy_acai_core の環境を、PPO 学習で扱いやすい形に包むラッパー。"""

    def __init__(
        self,
        toy_acai_core,
        max_steps: int,
        render: bool = False,
        module_dir: Optional[Path] = None,
        render_interval: float = RENDER_INTERVAL,
        random_start_positions: bool = True,
        learner_count: Optional[int] = None,
        opponent_count: Optional[int] = None,
        rng: Optional[object] = None,
    ):
        self.core = toy_acai_core
        self.max_steps = max_steps
        self.opponent = RuleBasedOpponent()
        self.step_count = 0
        self.render = render
        self.module_dir = module_dir
        self.render_interval = render_interval
        self.random_start_positions = bool(random_start_positions)
        self.learner_count = self._clamp_learner_count(learner_count)
        self.opponent_count = self._clamp_learner_count(opponent_count)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.env = self._make_env()
        self._recovery_env = None
        self.last_obs = None
        self._learner_position_history = []
        self._recovery_history: List[_RecoveryHistoryEntry] = []
        self._pending_recovery_teacher_examples: List[RecoveryTeacherExample] = []

    def _clamp_learner_count(self, learner_count: Optional[int]) -> int:
        team_count = int(self.core.TEAM_FIGHTER_COUNT)
        if learner_count is None:
            return team_count
        return int(np.clip(int(learner_count), 1, team_count))

    def _make_env(self):
        # render=True のときだけ Siv3D の描画リソース設定を有効にする。
        if self.render:
            if self.module_dir is not None and (self.module_dir / "resources").exists():
                os.chdir(self.module_dir)
        env_kwargs = {
            "render": self.render,
            "render_width": int(1920 * 0.3),
            "render_height": int(1080 * 0.3),
            "render_interval": self.render_interval,
            "active_blue_count": self.learner_count,
            "active_red_count": self.opponent_count,
        }
        return self.core.BattlefieldEnv(**env_kwargs)

    def reset(self) -> np.ndarray:
        # シミュレータを初期化し、生の状態ではなく学習用の観測ベクトルを返す。
        self.step_count = 0
        self.last_obs = self.env.reset()
        self._apply_random_start_positions()
        self._reset_movement_history()
        self._reset_recovery_history()
        return build_agent_observations(
            self.last_obs,
            active_learner_count=self.learner_count,
        )

    def _apply_random_start_positions(self) -> None:
        if not self.random_start_positions:
            return
        if not hasattr(self.env, "set_fighter_poses"):
            raise RuntimeError(
                "toy_acai_core.BattlefieldEnv.set_fighter_poses is required for random start positions"
            )
        poses = self._sample_random_start_poses(self.last_obs)
        self.last_obs = self.env.set_fighter_poses(poses)

    def _sample_random_start_poses(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        fighters = np.asarray(obs["fighters"], dtype=np.float64)
        field_w = float(obs["battlefield"][2])
        field_h = float(obs["battlefield"][3])
        base_poses = np.ascontiguousarray(fighters[:, 2:5], dtype=np.float64)

        learn_indices = _team_indices(fighters, TEAM_LEARN, self.learner_count)
        rule_indices = _team_indices(fighters, TEAM_RULE, self.opponent_count)
        learn_count = len(learn_indices)
        rule_count = len(rule_indices)

        if learn_count == 0 or rule_count == 0:
            poses = base_poses.copy()
            for team_id, active_count in (
                (TEAM_LEARN, self.learner_count),
                (TEAM_RULE, self.opponent_count),
            ):
                indices = _team_indices(fighters, team_id, active_count)
                team_poses = self._sample_team_start_poses(
                    team_id,
                    len(indices),
                    field_w,
                    field_h,
                )
                for pose, fighter_idx in zip(team_poses, indices):
                    poses[int(fighter_idx), :] = pose
            return poses

        for _ in range(RANDOM_START_MAX_ATTEMPTS):
            poses = base_poses.copy()
            learn_poses = self._sample_team_start_poses(
                TEAM_LEARN,
                learn_count,
                field_w,
                field_h,
            )
            rule_poses = self._sample_team_start_poses(
                TEAM_RULE,
                rule_count,
                field_w,
                field_h,
            )
            if (
                _min_pairwise_distance(learn_poses[:, :2], rule_poses[:, :2])
                < MIN_TEAM_START_DISTANCE
            ):
                continue

            for pose, fighter_idx in zip(learn_poses, learn_indices):
                poses[int(fighter_idx), :] = pose
            for pose, fighter_idx in zip(rule_poses, rule_indices):
                poses[int(fighter_idx), :] = pose
            return poses

        for pose, fighter_idx in zip(learn_poses, learn_indices):
            poses[int(fighter_idx), :] = pose
        for pose, fighter_idx in zip(rule_poses, rule_indices):
            poses[int(fighter_idx), :] = pose
        return poses

    def _sample_team_start_poses(
        self,
        team_id: int,
        count: int,
        field_w: float,
        field_h: float,
    ) -> np.ndarray:
        if count <= 0:
            return np.zeros((0, 3), dtype=np.float64)

        x_low_frac, x_high_frac = RANDOM_START_X_RANGE
        y_low_frac, y_high_frac = RANDOM_START_Y_RANGE
        xs = self.rng.uniform(x_low_frac * field_w, x_high_frac * field_w, size=count)

        y_low = y_low_frac * field_h
        y_high = y_high_frac * field_h
        slot_height = (y_high - y_low) / count
        slot_order = np.asarray(self.rng.permutation(count), dtype=np.float64)
        ys = y_low + (slot_order + self.rng.uniform(0.2, 0.8, size=count)) * slot_height

        base_yaw = 0.0 if team_id == TEAM_LEARN else math.pi
        yaws = base_yaw + self.rng.uniform(
            -RANDOM_START_YAW_JITTER,
            RANDOM_START_YAW_JITTER,
            size=count,
        )
        return np.stack([xs, ys, yaws], axis=1).astype(np.float64)

    def pop_recovery_teacher_examples(self) -> List[RecoveryTeacherExample]:
        examples = self._pending_recovery_teacher_examples
        self._pending_recovery_teacher_examples = []
        return examples

    def _reset_recovery_history(self) -> None:
        self._recovery_history = []
        self._pending_recovery_teacher_examples = []
        self._remember_recovery_state(self.last_obs)

    def _recovery_snapshot_available(self) -> bool:
        return hasattr(self.env, "snapshot") and hasattr(self.env, "restore_snapshot")

    def _remember_recovery_state(self, obs: Dict[str, object]) -> None:
        if obs is None or not self._recovery_snapshot_available():
            return
        self._recovery_history.append(
            _RecoveryHistoryEntry(
                step_count=self.step_count,
                snapshot=self.env.snapshot(),
                obs=_copy_observation(obs),
            )
        )
        max_history = RECOVERY_ROLLBACK_STEPS + RECOVERY_ROLLOUT_STEPS + 2
        if len(self._recovery_history) > max_history:
            self._recovery_history = self._recovery_history[-max_history:]

    def _recovery_history_entry_for_step(
        self, step_count: int
    ) -> Optional[_RecoveryHistoryEntry]:
        for entry in reversed(self._recovery_history):
            if entry.step_count == step_count:
                return entry
        return None

    def _make_recovery_env(self):
        if self._recovery_env is None:
            self._recovery_env = self.core.BattlefieldEnv(
                render=False,
                active_blue_count=self.learner_count,
                active_red_count=self.opponent_count,
            )
        return self._recovery_env

    def step(self, learner_actions: np.ndarray) -> StepResult:
        if self.last_obs is None:
            raise RuntimeError("reset() must be called before step()")

        # まず赤チームのルールベース行動を全機ぶん作り、
        # そこへ学習対象である青チームの行動を上書きする。
        actions = self.opponent.actions(self.last_obs, self.core.FIGHTER_COUNT)
        learner_indices = _team_indices(
            np.asarray(self.last_obs["fighters"]),
            TEAM_LEARN,
            self.learner_count,
        )
        applied_learner_actions = np.asarray(learner_actions, dtype=np.float64)
        for row, fighter_idx in enumerate(learner_indices):
            if row >= len(applied_learner_actions):
                break
            actions[fighter_idx, :] = applied_learner_actions[row, :]

        next_obs = self.env.step(actions)
        self.step_count += 1
        self._remember_recovery_state(next_obs)
        low_movement_distances_1s = self._record_learner_positions(next_obs)

        # 終了条件は「どちらかが全滅」または「最大ステップ到達」。
        fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        blue_alive = self._team_alive(fighters, TEAM_LEARN, self.learner_count)
        red_alive = self._team_alive(fighters, TEAM_RULE, self.opponent_count)
        done = bool(
            blue_alive == 0 or red_alive == 0 or self.step_count >= self.max_steps
        )
        agent_rewards, auxiliary_info = auxiliary_agent_rewards(
            next_obs,
            previous_obs=self.last_obs,
            low_movement_distances_1s=low_movement_distances_1s,
            learner_count=self.learner_count,
        )
        recovery_info = self._maybe_generate_recovery_teachers(self.last_obs, next_obs)
        info = {
            "blue_alive": float(blue_alive),
            "red_alive": float(red_alive),
            "outcome": 0.0,
            **auxiliary_info,
            **recovery_info,
        }
        if done:
            # 終了時だけ勝敗に応じた大きな報酬を足す。
            score = terminal_score(
                blue_alive=blue_alive,
                red_alive=red_alive,
                episode_steps=self.step_count,
                max_steps=self.max_steps,
                team_size=self.learner_count,
                opponent_team_size=self.opponent_count,
            )
            agent_rewards += score
            info["terminal_score"] = score
            if red_alive == 0:
                info["outcome"] = 1.0
            elif blue_alive == 0 and red_alive > 0:
                info["outcome"] = -1.0

        self.last_obs = next_obs
        info["reward_mean"] = float(np.mean(agent_rewards))
        return StepResult(
            build_agent_observations(
                next_obs,
                active_learner_count=self.learner_count,
            ),
            agent_rewards.astype(np.float32),
            done,
            info,
        )

    def _maybe_generate_recovery_teachers(
        self,
        previous_obs: Dict[str, object],
        next_obs: Dict[str, object],
    ) -> Dict[str, float]:
        info = {
            "recovery_teacher_attempts": 0.0,
            "recovery_teacher_successes": 0.0,
            "recovery_teacher_examples": 0.0,
            "recovery_teacher_candidates": 0.0,
            "recovery_teacher_best_score": 0.0,
        }
        killed = self._missile_killed_learner_rows(previous_obs, next_obs)
        if not killed:
            return info
        if not self._recovery_snapshot_available():
            return info

        rollback_step = self.step_count - RECOVERY_ROLLBACK_STEPS
        if rollback_step < 0:
            return info
        rollback_entry = self._recovery_history_entry_for_step(rollback_step)
        if rollback_entry is None:
            return info

        for agent_id, fighter_idx in killed:
            info["recovery_teacher_attempts"] += 1.0
            result = self._search_recovery_teacher(
                rollback_entry,
                agent_id=agent_id,
                fighter_idx=fighter_idx,
            )
            info["recovery_teacher_candidates"] += float(result["candidates"])
            if not result["examples"]:
                continue
            self._pending_recovery_teacher_examples.extend(result["examples"])
            info["recovery_teacher_successes"] += 1.0
            info["recovery_teacher_examples"] += float(len(result["examples"]))
            info["recovery_teacher_best_score"] = max(
                info["recovery_teacher_best_score"],
                float(result["score"]),
            )
        return info

    def _missile_killed_learner_rows(
        self,
        previous_obs: Dict[str, object],
        next_obs: Dict[str, object],
    ) -> List[Tuple[int, int]]:
        previous_fighters = np.asarray(previous_obs["fighters"], dtype=np.float64)
        next_fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        hit_events = np.asarray(
            next_obs.get("hit_events", np.zeros((0, 4), dtype=np.float64)),
            dtype=np.float64,
        )
        if hit_events.ndim == 1:
            hit_events = hit_events.reshape((-1, 4))
        hit_targets = {
            int(event[2])
            for event in hit_events
            if int(event[3]) == TEAM_LEARN
        }
        if not hit_targets:
            return []

        learner_indices = _team_indices(
            previous_fighters,
            TEAM_LEARN,
            self.learner_count,
        )
        killed = []
        for agent_id, fighter_idx in enumerate(learner_indices):
            fighter_idx = int(fighter_idx)
            if fighter_idx not in hit_targets:
                continue
            if previous_fighters[fighter_idx, 6] > 0.0 and next_fighters[fighter_idx, 6] <= 0.0:
                killed.append((agent_id, fighter_idx))
        return killed

    def _search_recovery_teacher(
        self,
        rollback_entry: _RecoveryHistoryEntry,
        *,
        agent_id: int,
        fighter_idx: int,
    ) -> Dict[str, object]:
        best_score = None
        best_examples: List[RecoveryTeacherExample] = []
        candidates = 0

        for segment_actions in itertools.product(
            RECOVERY_EXTREME_ACTIONS,
            repeat=RECOVERY_SEGMENT_COUNT,
        ):
            candidates += 1
            result = self._rollout_recovery_candidate(
                rollback_entry,
                agent_id=agent_id,
                fighter_idx=fighter_idx,
                segment_actions=segment_actions,
            )
            if result is None:
                continue
            score_tuple, examples = result
            if best_score is None or score_tuple > best_score:
                best_score = score_tuple
                best_examples = examples

        return {
            "candidates": candidates,
            "score": 0.0 if best_score is None else float(best_score[0]),
            "examples": best_examples,
        }

    def _rollout_recovery_candidate(
        self,
        rollback_entry: _RecoveryHistoryEntry,
        *,
        agent_id: int,
        fighter_idx: int,
        segment_actions: Tuple[np.ndarray, ...],
    ) -> Optional[Tuple[Tuple[float, float, float, float], List[RecoveryTeacherExample]]]:
        recovery_env = self._make_recovery_env()
        if not hasattr(recovery_env, "restore_snapshot"):
            return None
        obs = recovery_env.restore_snapshot(rollback_entry.snapshot)
        examples: List[RecoveryTeacherExample] = []
        min_missile_distance = self._nearest_hostile_missile_distance(obs, fighter_idx)

        for step_idx in range(RECOVERY_ROLLOUT_STEPS):
            action = segment_actions[step_idx // RECOVERY_SEGMENT_STEPS]
            if step_idx < RECOVERY_TEACHER_STEPS:
                agent_obs = build_agent_observations(
                    obs,
                    active_learner_count=self.learner_count,
                )
                examples.append(
                    RecoveryTeacherExample(
                        agent_id=agent_id,
                        observation=np.asarray(agent_obs[agent_id], dtype=np.float32).copy(),
                        action=np.asarray(action, dtype=np.float32).copy(),
                    )
                )

            actions = self.opponent.actions(obs, self.core.FIGHTER_COUNT)
            learner_indices = _team_indices(
                np.asarray(obs["fighters"], dtype=np.float64),
                TEAM_LEARN,
                self.learner_count,
            )
            for learner_idx in learner_indices:
                actions[int(learner_idx), :] = 0.0
            actions[fighter_idx, :] = action

            obs = recovery_env.step(actions)
            min_missile_distance = min(
                min_missile_distance,
                self._nearest_hostile_missile_distance(obs, fighter_idx),
            )
            fighters = np.asarray(obs["fighters"], dtype=np.float64)
            if fighters[fighter_idx, 6] <= 0.0:
                return None

        score = self._recovery_score_tuple(
            obs,
            fighter_idx=fighter_idx,
            min_missile_distance=min_missile_distance,
        )
        return score, examples

    def _nearest_hostile_missile_distance(
        self,
        obs: Dict[str, object],
        fighter_idx: int,
    ) -> float:
        fighters = np.asarray(obs["fighters"], dtype=np.float64)
        missiles = np.asarray(obs["missiles"], dtype=np.float64)
        if missiles.ndim == 1:
            missiles = missiles.reshape((-1, MISSILE_COLUMN_COUNT))
        battlefield = obs["battlefield"]
        diag = max(math.hypot(float(battlefield[2]), float(battlefield[3])), 1e-6)
        if len(missiles) == 0:
            return diag
        fighter = fighters[fighter_idx]
        hostile = missiles[missiles[:, MISSILE_TEAM_COLUMN] != fighter[0]]
        if len(hostile) == 0:
            return diag
        deltas = hostile[:, 0:2] - fighter[2:4]
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        return float(np.clip(np.min(distances), 0.0, diag))

    def _recovery_score_tuple(
        self,
        obs: Dict[str, object],
        *,
        fighter_idx: int,
        min_missile_distance: float,
    ) -> Tuple[float, float, float, float]:
        fighters = np.asarray(obs["fighters"], dtype=np.float64)
        final_missile_distance = self._nearest_hostile_missile_distance(obs, fighter_idx)
        blue_alive = self._team_alive(fighters, TEAM_LEARN, self.learner_count)
        red_alive = self._team_alive(fighters, TEAM_RULE, self.opponent_count)
        # 生存できた候補同士は、ミサイルからの最小距離、最終距離、場外時間、戦況で順に比べる。
        return (
            float(min_missile_distance),
            float(final_missile_distance),
            -float(fighters[fighter_idx, 8]),
            float(blue_alive - red_alive),
        )

    def take_render_frame(self):
        if not self.render or not hasattr(self.env, "take_render_frame"):
            return None
        return self.env.take_render_frame()

    def _current_learner_positions(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        fighters = np.asarray(obs["fighters"], dtype=np.float64)
        learner_indices = _team_indices(fighters, TEAM_LEARN, self.learner_count)
        return np.asarray(fighters[learner_indices, 2:4], dtype=np.float64).copy()

    def _reset_movement_history(self) -> None:
        self._learner_position_history = [
            self._current_learner_positions(self.last_obs)
        ]

    def _record_learner_positions(
        self, obs: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        self._learner_position_history.append(self._current_learner_positions(obs))
        max_history = LOW_MOVEMENT_WINDOW_STEPS + 1
        if len(self._learner_position_history) > max_history:
            self._learner_position_history = self._learner_position_history[
                -max_history:
            ]
        if len(self._learner_position_history) < max_history:
            return None
        return np.linalg.norm(
            self._learner_position_history[-1] - self._learner_position_history[0],
            axis=1,
        )

    @staticmethod
    def _team_alive(
        fighters: np.ndarray,
        team_id: int,
        active_count: Optional[int] = None,
    ) -> int:
        team_indices = _team_indices(fighters, team_id, active_count)
        return int(np.sum(fighters[team_indices, 6] > 0.0))
