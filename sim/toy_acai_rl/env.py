import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


TEAM_LEARN = 0
TEAM_RULE = 1
MAX_TRACKED_MISSILES = 8
MAX_SPEED = 360.0
OUT_OF_BOUNDS_DEATH_TIME = 3.0


def add_default_module_paths(repo_root: Path, module_dir: Optional[Path] = None) -> None:
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
        blue_indices = np.where((fighters[:, 0] == TEAM_LEARN) & alive)[0]

        for i, fighter in enumerate(fighters):
            if int(fighter[0]) != self.team_id or fighter[6] <= 0.0:
                continue

            actions[i, 0] = 0.55
            if len(blue_indices) == 0:
                actions[i, 1] = _edge_turn(fighter, field_w, field_h)
                continue

            deltas = fighters[blue_indices, 2:4] - fighter[2:4]
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


def build_agent_observations(obs: Dict[str, np.ndarray], learner_team: int = TEAM_LEARN) -> np.ndarray:
    fighters = np.asarray(obs["fighters"], dtype=np.float64)
    missiles = np.asarray(obs["missiles"], dtype=np.float64)
    field_w = float(obs["battlefield"][2])
    field_h = float(obs["battlefield"][3])
    diag = math.hypot(field_w, field_h)

    agent_indices = np.where(fighters[:, 0] == learner_team)[0]
    all_obs = []
    for agent_idx in agent_indices:
        fighter = fighters[agent_idx]
        features = []
        x = float(fighter[2])
        y = float(fighter[3])
        yaw = float(fighter[4])
        features.extend(
            [
                x / field_w * 2.0 - 1.0,
                y / field_h * 2.0 - 1.0,
                math.cos(yaw),
                math.sin(yaw),
                float(fighter[5]) / MAX_SPEED,
                float(fighter[6]),
                float(fighter[7]),
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
            features.extend(
                [
                    float(rel[0]) / field_w,
                    float(rel[1]) / field_h,
                    math.cos(float(other[4])),
                    math.sin(float(other[4])),
                    float(other[5]) / MAX_SPEED,
                    float(other[6]),
                    1.0 if int(other[0]) == learner_team else -1.0,
                ]
            )

        missile_features = []
        for missile in missiles:
            rel = missile[0:2] - fighter[2:4]
            distance = math.hypot(float(rel[0]), float(rel[1]))
            missile_features.append((distance, missile, rel))
        missile_features.sort(key=lambda item: item[0])
        for _, missile, rel in missile_features[:MAX_TRACKED_MISSILES]:
            features.extend(
                [
                    float(rel[0]) / field_w,
                    float(rel[1]) / field_h,
                    math.cos(float(missile[2])),
                    math.sin(float(missile[2])),
                    float(missile[3]) / MAX_SPEED,
                    float(missile[4]) / 6.0,
                    1.0 if int(missile[6]) == learner_team else -1.0,
                    1.0 if int(missile[7]) == int(agent_idx) else 0.0,
                ]
            )
        for _ in range(MAX_TRACKED_MISSILES - len(missile_features[:MAX_TRACKED_MISSILES])):
            features.extend([0.0] * 8)

        all_obs.append(features)

    return np.asarray(all_obs, dtype=np.float32)


def observation_dim(toy_acai_core) -> int:
    env = toy_acai_core.BattlefieldEnv()
    obs = env.reset()
    return int(build_agent_observations(obs).shape[1])


@dataclass
class StepResult:
    observations: np.ndarray
    rewards: np.ndarray
    done: bool
    info: Dict[str, float]


class ToyAcaiPPOEnv:
    def __init__(
        self,
        toy_acai_core,
        max_steps: int,
        render: bool = False,
        gif_path: Optional[Path] = None,
        module_dir: Optional[Path] = None,
    ):
        self.core = toy_acai_core
        self.max_steps = max_steps
        self.opponent = RuleBasedOpponent()
        self.step_count = 0
        self.render = render
        self.gif_path = gif_path
        self.module_dir = module_dir
        self.env = self._make_env()
        self.last_obs = None
        self.prev_blue_alive = 0
        self.prev_red_alive = 0

    def _make_env(self):
        if self.render and self.gif_path is not None:
            self.gif_path.parent.mkdir(parents=True, exist_ok=True)
            if self.module_dir is not None and (self.module_dir / "resources").exists():
                os.chdir(self.module_dir)
        return self.core.BattlefieldEnv(
            render=self.render,
            gif_path=str(self.gif_path) if self.render and self.gif_path is not None else "",
        )

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.last_obs = self.env.reset()
        fighters = np.asarray(self.last_obs["fighters"], dtype=np.float64)
        self.prev_blue_alive = self._team_alive(fighters, TEAM_LEARN)
        self.prev_red_alive = self._team_alive(fighters, TEAM_RULE)
        return build_agent_observations(self.last_obs)

    def step(self, learner_actions: np.ndarray) -> StepResult:
        if self.last_obs is None:
            raise RuntimeError("reset() must be called before step()")

        actions = self.opponent.actions(self.last_obs, self.core.FIGHTER_COUNT)
        learner_indices = np.where(np.asarray(self.last_obs["fighters"])[:, 0] == TEAM_LEARN)[0]
        for row, fighter_idx in enumerate(learner_indices):
            actions[fighter_idx, :] = learner_actions[row, :]

        next_obs = self.env.step(actions)
        self.step_count += 1

        reward, info = self._reward(self.last_obs, next_obs)
        self.last_obs = next_obs
        done = bool(info["blue_alive"] == 0 or info["red_alive"] == 0 or self.step_count >= self.max_steps)
        if done:
            if info["red_alive"] == 0 and info["blue_alive"] > 0:
                reward += 20.0
                info["outcome"] = 1.0
            elif info["blue_alive"] == 0 and info["red_alive"] > 0:
                reward -= 20.0
                info["outcome"] = -1.0
            else:
                info["outcome"] = 0.0

        agent_rewards = np.full((self.core.TEAM_FIGHTER_COUNT,), reward, dtype=np.float32)
        return StepResult(build_agent_observations(next_obs), agent_rewards, done, info)

    def close(self) -> None:
        if self.render:
            self.env.close_gif()

    def _reward(self, prev_obs: Dict[str, np.ndarray], next_obs: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        prev_fighters = np.asarray(prev_obs["fighters"], dtype=np.float64)
        next_fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        blue_alive = self._team_alive(next_fighters, TEAM_LEARN)
        red_alive = self._team_alive(next_fighters, TEAM_RULE)
        red_losses = self.prev_red_alive - red_alive
        blue_losses = self.prev_blue_alive - blue_alive
        self.prev_blue_alive = blue_alive
        self.prev_red_alive = red_alive

        reward = float(red_losses * 5.0 - blue_losses * 5.0)
        reward -= 0.002
        reward += self._aim_bonus(next_fighters)
        reward -= self._oob_penalty(next_fighters)

        return reward, {
            "blue_alive": float(blue_alive),
            "red_alive": float(red_alive),
            "red_losses": float(red_losses),
            "blue_losses": float(blue_losses),
            "outcome": 0.0,
        }

    @staticmethod
    def _team_alive(fighters: np.ndarray, team_id: int) -> int:
        return int(np.sum((fighters[:, 0] == team_id) & (fighters[:, 6] > 0.0)))

    @staticmethod
    def _aim_bonus(fighters: np.ndarray) -> float:
        bonus = 0.0
        blue = fighters[(fighters[:, 0] == TEAM_LEARN) & (fighters[:, 6] > 0.0)]
        red = fighters[(fighters[:, 0] == TEAM_RULE) & (fighters[:, 6] > 0.0)]
        for fighter in blue:
            if len(red) == 0:
                continue
            deltas = red[:, 2:4] - fighter[2:4]
            target = deltas[int(np.argmin(np.sum(deltas * deltas, axis=1)))]
            desired = math.atan2(float(target[1]), float(target[0]))
            if abs(_angle_delta(desired, float(fighter[4]))) < 0.45:
                bonus += 0.01
        return bonus

    @staticmethod
    def _oob_penalty(fighters: np.ndarray) -> float:
        blue = fighters[(fighters[:, 0] == TEAM_LEARN) & (fighters[:, 6] > 0.0)]
        return float(np.sum(np.clip(blue[:, 8] / OUT_OF_BOUNDS_DEATH_TIME, 0.0, 1.0)) * 0.02)
