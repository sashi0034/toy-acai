import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

TEAM_LEARN = 0
TEAM_RULE = 1
# 観測に入れるミサイル数を固定する。ニューラルネットは入力長が固定である必要があるため、
# 足りない分は 0 で埋める。
MAX_TRACKED_MISSILES = 8
MISSILE_OBS_FEATURES = 7
MAX_SPEED = 360.0
RENDER_INTERVAL = 0.1
AUX_KILL_REWARD = 1.0
AUX_DEATH_PENALTY = 5.0
# AUX_SURVIVAL_REWARD_PER_STEP = 0.0015 # 生存報酬は一旦無効化する。必要になったら再度有効化する。
AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP = 0.03
AUX_INCOMING_MISSILE_PENALTY = 0.10
AUX_INCOMING_MISSILE_RADIUS = 100.0
AUX_MISSILE_FIRE_REWARD = 0.03
# AUX_MOVEMENT_REWARD_PER_DISTANCE = 0.10


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
            seeker_half_angle = 0.85
            in_fire_arc = 1.0 if abs(bearing_delta) <= seeker_half_angle else 0.0
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
            if int(missile[6]) == learner_team:
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


def auxiliary_agent_rewards(
    obs: Dict[str, np.ndarray],
    previous_obs: Optional[Dict[str, np.ndarray]] = None,
    learner_team: int = TEAM_LEARN,
    opponent_team: int = TEAM_RULE,
    kill_reward: float = AUX_KILL_REWARD,
    death_penalty: float = AUX_DEATH_PENALTY,
    out_of_bounds_penalty_per_step: float = AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP,
    incoming_missile_penalty: float = AUX_INCOMING_MISSILE_PENALTY,
    incoming_missile_radius: float = AUX_INCOMING_MISSILE_RADIUS,
    missile_fire_reward: float = AUX_MISSILE_FIRE_REWARD,
    # movement_reward_per_distance: float = AUX_MOVEMENT_REWARD_PER_DISTANCE,
    learner_count: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # 毎ステップ与える補助報酬。終端報酬だけだと「何が良かったか」が遠すぎるため、
    # 場外・ミサイル・撃墜・損失を小さな手がかりとして追加して学習を助ける。
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

    incoming_missile_penalty_total = 0.0
    missile_radius = max(float(incoming_missile_radius), 1e-6)
    if missiles.ndim == 1:
        missiles = missiles.reshape((-1, 8))
    enemy_missiles = (
        missiles[missiles[:, 6] != float(learner_team)]
        if len(missiles)
        else missiles
    )
    for agent_idx, fighter_idx in enumerate(learner_indices):
        if not alive[agent_idx]:
            continue
        fighter = fighters[int(fighter_idx)]
        fighter_pos = fighter[2:4]
        fighter_penalty = 0.0
        for missile in enemy_missiles:
            rel = missile[0:2] - fighter_pos
            distance = math.hypot(float(rel[0]), float(rel[1]))
            if distance <= 1e-6 or distance > missile_radius:
                continue
            missile_yaw = float(missile[2])
            missile_forward = np.array(
                [math.cos(missile_yaw), math.sin(missile_yaw)], dtype=np.float64
            )
            incoming_alignment = -float(np.dot(rel / distance, missile_forward))
            targets_fighter = int(missile[7]) == int(fighter_idx)
            if incoming_alignment <= 0.5 and not targets_fighter:
                continue
            near_factor = 1.0 - distance / missile_radius
            incoming_factor = max(incoming_alignment, 0.5 if targets_fighter else 0.0)
            fighter_penalty += (
                float(incoming_missile_penalty)
                * near_factor
                * float(np.clip(incoming_factor, 0.0, 1.0))
            )
        if fighter_penalty > 0.0:
            rewards[agent_idx] -= float(fighter_penalty)
            incoming_missile_penalty_total += fighter_penalty

    movement_reward = 0.0
    mean_movement_distance = 0.0
    missile_fire_reward_total = 0.0
    if previous_obs is not None:
        previous_fighters = np.asarray(previous_obs["fighters"], dtype=np.float64)
        previous_alive = previous_fighters[learner_indices, 6] > 0.0
        previous_cooldown = previous_fighters[learner_indices, 7]
        current_cooldown = learner_fighters[:, 7]
        launched = (
            alive
            & previous_alive
            & (current_cooldown > previous_cooldown + 1e-6)
        )
        fire_rewards = launched.astype(np.float64) * float(missile_fire_reward)
        rewards += fire_rewards.astype(np.float32)
        missile_fire_reward_total = float(np.sum(fire_rewards))

        movement_tracked = alive & previous_alive
        movement_eligible = movement_tracked & in_bounds
        movement_distance = np.linalg.norm(
            fighters[learner_indices, 2:4] - previous_fighters[learner_indices, 2:4],
            axis=1,
        )
        # ノロノロ対策として、戦場内で実際に移動した距離をごく小さく加点する。
        field_w = float(obs["battlefield"][2])
        field_h = float(obs["battlefield"][3])
        reward_distance = max(math.hypot(field_w, field_h), 1e-6)
        clipped_movement = np.clip(movement_distance / reward_distance, 0.0, 1.0)
        movement_rewards = clipped_movement * 0.10
        movement_rewards[~movement_eligible] = 0.0
        rewards += movement_rewards.astype(np.float32)
        movement_reward = float(np.sum(movement_rewards))
        if np.any(movement_tracked):
            mean_movement_distance = float(np.mean(movement_distance[movement_tracked]))

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
        if agent_idx is None or shooter_team != learner_team or target_team != opponent_team:
            continue
        # 撃墜した本人には大きめの報酬を与える。
        rewards[agent_idx] += float(kill_reward)
        blue_kills += 1
    kill_reward_total = float(blue_kills * kill_reward)
    blue_losses = 0
    if previous_obs is not None:
        for fighter_idx in learner_indices:
            fighter_idx = int(fighter_idx)
            if previous_fighters[fighter_idx, 6] <= 0.0 or fighters[fighter_idx, 6] > 0.0:
                continue
            agent_idx = fighter_to_agent[fighter_idx]
            # 前ステップでは生存、今ステップでは非生存なら、その機が撃墜されたとみなす。
            rewards[agent_idx] -= float(death_penalty)
            blue_losses += 1

    death_penalty_total = float(blue_losses * death_penalty)
    info = {
        "survival_reward": survival_reward,
        "out_of_bounds_penalty": out_of_bounds_penalty_total,
        "incoming_missile_penalty": incoming_missile_penalty_total,
        "movement_reward": movement_reward,
        "mean_movement_distance": mean_movement_distance,
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


class ToyAcaiPPOEnv:
    """toy_acai_core の環境を、PPO 学習で扱いやすい形に包むラッパー。"""

    def __init__(
        self,
        toy_acai_core,
        max_steps: int,
        render: bool = False,
        module_dir: Optional[Path] = None,
        render_interval: float = RENDER_INTERVAL,
        random_start_steps: int = 0,
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
        self.random_start_steps = random_start_steps
        self.learner_count = self._clamp_learner_count(learner_count)
        self.opponent_count = self._clamp_learner_count(opponent_count)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.env = self._make_env()
        self.last_obs = None

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
        self._apply_random_start()
        return build_agent_observations(
            self.last_obs,
            active_learner_count=self.learner_count,
        )

    def _apply_random_start(self) -> None:
        # 毎回まったく同じ初期配置から始めると過学習しやすい。
        # 数ステップだけランダムに動かして、開始状況にばらつきを作る。
        for _ in range(max(0, self.random_start_steps)):
            actions = np.zeros((self.core.FIGHTER_COUNT, 3), dtype=np.float64)
            actions[:, 0] = self.rng.uniform(0.15, 0.9, size=self.core.FIGHTER_COUNT)
            actions[:, 1] = self.rng.uniform(-1.0, 1.0, size=self.core.FIGHTER_COUNT)
            self.last_obs = self.env.step(actions)

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

        # 終了条件は「どちらかが全滅」または「最大ステップ到達」。
        fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        blue_alive = self._team_alive(fighters, TEAM_LEARN, self.learner_count)
        red_alive = self._team_alive(fighters, TEAM_RULE, self.opponent_count)
        done = bool(
            blue_alive == 0
            or red_alive == 0
            or self.step_count >= self.max_steps
        )
        agent_rewards, auxiliary_info = auxiliary_agent_rewards(
            next_obs,
            previous_obs=self.last_obs,
            learner_count=self.learner_count,
        )
        info = {
            "blue_alive": float(blue_alive),
            "red_alive": float(red_alive),
            "outcome": 0.0,
            **auxiliary_info,
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

    def take_render_frame(self):
        if not self.render or not hasattr(self.env, "take_render_frame"):
            return None
        return self.env.take_render_frame()

    @staticmethod
    def _team_alive(
        fighters: np.ndarray,
        team_id: int,
        active_count: Optional[int] = None,
    ) -> int:
        team_indices = _team_indices(fighters, team_id, active_count)
        return int(np.sum(fighters[team_indices, 6] > 0.0))
