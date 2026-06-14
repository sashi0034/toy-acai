import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

TEAM_LEARN = 0
TEAM_RULE = 1
MAX_TRACKED_MISSILES = 8
MISSILE_OBS_FEATURES = 14
MAX_SPEED = 360.0
OUT_OF_BOUNDS_DEATH_TIME = 3.0
RENDER_INTERVAL = 0.1


def add_default_module_paths(
    repo_root: Path, module_dir: Optional[Path] = None
) -> None:
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
    return math.remainder(target - current, math.tau)


def _alive(fighters: np.ndarray) -> np.ndarray:
    return fighters[:, 6] > 0.0


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


def build_agent_observations(
    obs: Dict[str, np.ndarray], learner_team: int = TEAM_LEARN
) -> np.ndarray:
    fighters = np.asarray(obs["fighters"], dtype=np.float64)
    missiles = np.asarray(obs["missiles"], dtype=np.float64)
    field_w = float(obs["battlefield"][2])
    field_h = float(obs["battlefield"][3])
    diag = max(math.hypot(field_w, field_h), 1e-6)

    agent_indices = np.where(fighters[:, 0] == learner_team)[0]
    all_obs = []
    for agent_idx in agent_indices:
        fighter = fighters[agent_idx]
        features = []
        x = float(fighter[2])
        y = float(fighter[3])
        yaw = float(fighter[4])
        speed = float(fighter[5])
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        right = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)
        velocity = forward * speed
        features.extend(
            [
                x / field_w * 2.0 - 1.0,
                y / field_h * 2.0 - 1.0,
                float(forward[0]),
                float(forward[1]),
                speed / MAX_SPEED,
                float(fighter[6]),
                float(fighter[7]),
                1.0 if float(fighter[7]) <= 0.0 else 0.0,
                float(fighter[8]) / OUT_OF_BOUNDS_DEATH_TIME,
                x / field_w,
                (field_w - x) / field_w,
                y / field_h,
                (field_h - y) / field_h,
            ]
        )

        others = [i for i in range(len(fighters)) if i != agent_idx]
        others.sort(key=lambda i: (fighters[i, 0] == learner_team, i))
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
            features.extend(
                [
                    float(rel[0]) / field_w,
                    float(rel[1]) / field_h,
                    float(np.dot(rel, forward)) / diag,
                    float(np.dot(rel, right)) / diag,
                    distance / diag,
                    math.cos(bearing_delta),
                    math.sin(bearing_delta),
                    math.cos(_angle_delta(other_yaw, yaw)),
                    math.sin(_angle_delta(other_yaw, yaw)),
                    other_speed / MAX_SPEED,
                    float(other[6]),
                    1.0 if int(other[0]) == learner_team else -1.0,
                    float(np.clip(closing, -2.0, 2.0)),
                ]
            )

        missile_features = []
        for missile in missiles:
            rel = missile[0:2] - fighter[2:4]
            distance = math.hypot(float(rel[0]), float(rel[1]))
            missile_features.append((distance, missile, rel))
        missile_features.sort(key=lambda item: item[0])
        for _, missile, rel in missile_features[:MAX_TRACKED_MISSILES]:
            distance = math.hypot(float(rel[0]), float(rel[1]))
            bearing = math.atan2(float(rel[1]), float(rel[0]))
            bearing_delta = _angle_delta(bearing, yaw)
            missile_yaw = float(missile[2])
            features.extend(
                [
                    float(rel[0]) / field_w,
                    float(rel[1]) / field_h,
                    float(np.dot(rel, forward)) / diag,
                    float(np.dot(rel, right)) / diag,
                    distance / diag,
                    math.cos(bearing_delta),
                    math.sin(bearing_delta),
                    math.cos(_angle_delta(missile_yaw, yaw)),
                    math.sin(_angle_delta(missile_yaw, yaw)),
                    float(missile[3]) / MAX_SPEED,
                    float(missile[4]) / 6.0,
                    float(missile[5]) / 1.1,
                    1.0 if int(missile[6]) == learner_team else -1.0,
                    1.0 if int(missile[7]) == int(agent_idx) else 0.0,
                ]
            )
        for _ in range(
            MAX_TRACKED_MISSILES - len(missile_features[:MAX_TRACKED_MISSILES])
        ):
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
) -> float:
    team_size = max(1, int(team_size))
    blue_alive_ratio = float(blue_alive) / team_size
    red_alive_ratio = float(red_alive) / team_size
    if red_alive == 0:
        time_bonus = 0.0
        if max_steps > 0:
            time_bonus = np.clip((max_steps - episode_steps) / max_steps, 0.0, 1.0)
        return float(1.0 + 0.1 * blue_alive_ratio + 0.01 * time_bonus)
    return float(-red_alive_ratio + 0.1 * blue_alive_ratio)


class StepResult:
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
    def __init__(
        self,
        toy_acai_core,
        max_steps: int,
        render: bool = False,
        gif_path: Optional[Path] = None,
        module_dir: Optional[Path] = None,
        render_interval: float = RENDER_INTERVAL,
        random_start_steps: int = 0,
        rng: Optional[object] = None,
    ):
        self.core = toy_acai_core
        self.max_steps = max_steps
        self.opponent = RuleBasedOpponent()
        self.step_count = 0
        self.render = render
        self.gif_path = gif_path
        self.module_dir = module_dir
        self.render_interval = render_interval
        self.random_start_steps = random_start_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.env = self._make_env()
        self.last_obs = None

    def _make_env(self):
        if self.render and self.gif_path is not None:
            self.gif_path.parent.mkdir(parents=True, exist_ok=True)
            if self.module_dir is not None and (self.module_dir / "resources").exists():
                os.chdir(self.module_dir)
        env_kwargs = {
            "render": self.render,
            "render_width": int(1920 * 0.3),
            "render_height": int(1080 * 0.3),
            "gif_path": (
                str(self.gif_path) if self.render and self.gif_path is not None else ""
            ),
            "render_interval": self.render_interval,
        }
        return self.core.BattlefieldEnv(**env_kwargs)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.last_obs = self.env.reset()
        self._apply_random_start()
        return build_agent_observations(self.last_obs)

    def _apply_random_start(self) -> None:
        for _ in range(max(0, self.random_start_steps)):
            actions = np.zeros((self.core.FIGHTER_COUNT, 3), dtype=np.float64)
            actions[:, 0] = self.rng.uniform(0.15, 0.9, size=self.core.FIGHTER_COUNT)
            actions[:, 1] = self.rng.uniform(-1.0, 1.0, size=self.core.FIGHTER_COUNT)
            self.last_obs = self.env.step(actions)

    def step(self, learner_actions: np.ndarray) -> StepResult:
        if self.last_obs is None:
            raise RuntimeError("reset() must be called before step()")

        actions = self.opponent.actions(self.last_obs, self.core.FIGHTER_COUNT)
        learner_indices = np.where(
            np.asarray(self.last_obs["fighters"])[:, 0] == TEAM_LEARN
        )[0]
        applied_learner_actions = np.asarray(learner_actions, dtype=np.float64)
        for row, fighter_idx in enumerate(learner_indices):
            if row >= len(applied_learner_actions):
                break
            actions[fighter_idx, :] = applied_learner_actions[row, :]

        next_obs = self.env.step(actions)
        self.step_count += 1

        fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        blue = fighters[fighters[:, 0] == TEAM_LEARN]
        blue_alive = self._team_alive(fighters, TEAM_LEARN)
        red_alive = self._team_alive(fighters, TEAM_RULE)
        done = bool(
            blue_alive == 0
            or red_alive == 0
            or self.step_count >= self.max_steps
        )
        agent_rewards = np.zeros((len(blue),), dtype=np.float32)
        info = {
            "blue_alive": float(blue_alive),
            "red_alive": float(red_alive),
            "outcome": 0.0,
        }
        if done:
            score = terminal_score(
                blue_alive=blue_alive,
                red_alive=red_alive,
                episode_steps=self.step_count,
                max_steps=self.max_steps,
                team_size=int(self.core.TEAM_FIGHTER_COUNT),
            )
            agent_rewards[:] = score
            info["terminal_score"] = score
            if red_alive == 0:
                info["outcome"] = 1.0
            elif blue_alive == 0 and red_alive > 0:
                info["outcome"] = -1.0

        self.last_obs = next_obs
        info["reward_mean"] = float(np.mean(agent_rewards))
        return StepResult(
            build_agent_observations(next_obs),
            agent_rewards.astype(np.float32),
            done,
            info,
        )

    def close(self) -> None:
        if self.render:
            self.env.close_gif()

    @staticmethod
    def _team_alive(fighters: np.ndarray, team_id: int) -> int:
        return int(np.sum((fighters[:, 0] == team_id) & (fighters[:, 6] > 0.0)))
