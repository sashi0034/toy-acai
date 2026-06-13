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
MISSILE_OBS_FEATURES = 14
MAX_SPEED = 360.0
OUT_OF_BOUNDS_DEATH_TIME = 3.0
MISSILE_SEEKER_HALF_ANGLE = 0.85
RENDER_INTERVAL = 0.1
STEP_PENALTY = 0.002
TEAM_KILL_REWARD = 5.0
WIN_REWARD = 20.0
SURVIVOR_ADVANTAGE_REWARD = 2.5
FIRE_SUCCESS_BONUS = 0.35
MISSED_FIRE_OPPORTUNITY_PENALTY = 0.04
BAD_FIRE_ATTEMPT_PENALTY = 0.02
COOLDOWN_FIRE_ATTEMPT_PENALTY = 0.003
MIN_LEARNER_ACCELERATION = 0.10
AIM_BONUS_SCALE = 0.006
ENGAGEMENT_CLOSING_BONUS_SCALE = 4.0
DISTANT_LOITER_PENALTY = 0.004
LOW_SPEED_PENALTY_SCALE = 0.003
TURN_PENALTY_SCALE = 0.0015


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
        render_interval: float = RENDER_INTERVAL,
        random_start_steps: int = 0,
        rng: Optional[np.random.Generator] = None,
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
        self.prev_blue_alive = 0
        self.prev_red_alive = 0
        self.episode_fire_attempts = 0
        self.episode_fire_opportunities = 0
        self.episode_fire_successes = 0
        self.episode_blocked_fire_attempts = 0
        self.episode_requested_fire_inputs = 0

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
        fighters = np.asarray(self.last_obs["fighters"], dtype=np.float64)
        self.prev_blue_alive = self._team_alive(fighters, TEAM_LEARN)
        self.prev_red_alive = self._team_alive(fighters, TEAM_RULE)
        self.episode_fire_attempts = 0
        self.episode_fire_opportunities = 0
        self.episode_fire_successes = 0
        self.episode_blocked_fire_attempts = 0
        self.episode_requested_fire_inputs = 0
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
        prev_fighters = np.asarray(self.last_obs["fighters"], dtype=np.float64)
        applied_learner_actions, action_info = self._apply_learner_fire_gate(
            prev_fighters, learner_actions
        )
        applied_learner_actions = self._apply_learner_motion_floor(
            prev_fighters, applied_learner_actions
        )
        for row, fighter_idx in enumerate(learner_indices):
            actions[fighter_idx, :] = applied_learner_actions[row, :]

        next_obs = self.env.step(actions)
        self.step_count += 1

        agent_rewards, info = self._reward(
            self.last_obs, next_obs, action_info, applied_learner_actions
        )
        self._accumulate_fire_info(info)
        self.last_obs = next_obs
        done = bool(
            info["blue_alive"] == 0
            or info["red_alive"] == 0
            or self.step_count >= self.max_steps
        )
        if done:
            if info["red_alive"] == 0 and info["blue_alive"] > 0:
                agent_rewards += WIN_REWARD
                info["outcome"] = 1.0
            elif info["blue_alive"] == 0 and info["red_alive"] > 0:
                agent_rewards -= WIN_REWARD
                info["outcome"] = -1.0
            else:
                agent_rewards += SURVIVOR_ADVANTAGE_REWARD * (
                    info["blue_alive"] - info["red_alive"]
                )
                info["outcome"] = 0.0

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

    def _reward(
        self,
        prev_obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action_info: Dict[str, object],
        learner_actions: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        prev_fighters = np.asarray(prev_obs["fighters"], dtype=np.float64)
        next_fighters = np.asarray(next_obs["fighters"], dtype=np.float64)
        blue_alive = self._team_alive(next_fighters, TEAM_LEARN)
        red_alive = self._team_alive(next_fighters, TEAM_RULE)
        red_losses = self.prev_red_alive - red_alive
        blue_losses = self.prev_blue_alive - blue_alive
        self.prev_blue_alive = blue_alive
        self.prev_red_alive = red_alive

        blue = next_fighters[next_fighters[:, 0] == TEAM_LEARN]
        agent_rewards = np.full((len(blue),), -STEP_PENALTY, dtype=np.float32)
        agent_rewards += float(
            red_losses * TEAM_KILL_REWARD - blue_losses * TEAM_KILL_REWARD
        )
        agent_rewards += self._agent_aim_bonus(next_fighters)
        agent_rewards += self._agent_engagement_bonus(prev_fighters, next_fighters)
        fire_bonus, fire_info = self._fire_feedback(action_info)
        agent_rewards += self._agent_fire_feedback(action_info, len(blue))
        agent_rewards -= self._agent_low_speed_penalty(next_fighters)
        agent_rewards -= self._agent_oob_penalty(next_fighters)
        agent_rewards -= self._turn_penalty(next_fighters, learner_actions)

        return agent_rewards, {
            "blue_alive": float(blue_alive),
            "red_alive": float(red_alive),
            "red_losses": float(red_losses),
            "blue_losses": float(blue_losses),
            "outcome": 0.0,
            "team_reward": float(
                red_losses * TEAM_KILL_REWARD - blue_losses * TEAM_KILL_REWARD
            ),
            "fire_reward": float(fire_bonus),
            **fire_info,
        }

    def _accumulate_fire_info(self, info: Dict[str, float]) -> None:
        self.episode_fire_attempts += int(info.get("fire_attempts", 0.0))
        self.episode_fire_opportunities += int(info.get("fire_opportunities", 0.0))
        self.episode_fire_successes += int(info.get("fire_successes", 0.0))
        self.episode_blocked_fire_attempts += int(
            info.get("blocked_fire_attempts", 0.0)
        )
        self.episode_requested_fire_inputs += int(
            info.get("requested_fire_inputs", 0.0)
        )
        info["episode_fire_attempts"] = float(self.episode_fire_attempts)
        info["episode_fire_opportunities"] = float(self.episode_fire_opportunities)
        info["episode_fire_successes"] = float(self.episode_fire_successes)
        info["episode_blocked_fire_attempts"] = float(
            self.episode_blocked_fire_attempts
        )
        info["episode_requested_fire_inputs"] = float(
            self.episode_requested_fire_inputs
        )

    @staticmethod
    def _team_alive(fighters: np.ndarray, team_id: int) -> int:
        return int(np.sum((fighters[:, 0] == team_id) & (fighters[:, 6] > 0.0)))

    @staticmethod
    def _agent_aim_bonus(fighters: np.ndarray) -> np.ndarray:
        blue = fighters[fighters[:, 0] == TEAM_LEARN]
        red = fighters[(fighters[:, 0] == TEAM_RULE) & (fighters[:, 6] > 0.0)]
        rewards = np.zeros((len(blue),), dtype=np.float32)
        if len(red) == 0:
            return rewards

        for row, fighter in enumerate(blue):
            if fighter[6] <= 0.0:
                continue
            deltas = red[:, 2:4] - fighter[2:4]
            target = deltas[int(np.argmin(np.sum(deltas * deltas, axis=1)))]
            desired = math.atan2(float(target[1]), float(target[0]))
            rewards[row] = AIM_BONUS_SCALE * math.cos(
                _angle_delta(desired, float(fighter[4]))
            )
        return rewards

    def _apply_learner_fire_gate(
        self, fighters: np.ndarray, learner_actions: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        applied = np.asarray(learner_actions, dtype=np.float64).copy()
        blue = fighters[(fighters[:, 0] == TEAM_LEARN)]
        red = fighters[(fighters[:, 0] == TEAM_RULE) & (fighters[:, 6] > 0.0)]
        requested_inputs = 0
        attempts = 0
        opportunities = 0
        blocked_attempts = 0
        cooldown_blocked_attempts = 0
        requested_by_agent = np.zeros((len(blue),), dtype=np.float32)
        attempts_by_agent = np.zeros((len(blue),), dtype=np.float32)
        opportunities_by_agent = np.zeros((len(blue),), dtype=np.float32)
        blocked_by_agent = np.zeros((len(blue),), dtype=np.float32)
        cooldown_blocked_by_agent = np.zeros((len(blue),), dtype=np.float32)

        for row, fighter in enumerate(blue):
            if row >= len(applied):
                continue

            requested = bool(applied[row, 2] >= 0.5)
            applied[row, 2] = 0.0
            if fighter[6] <= 0.0:
                continue

            has_target = self._has_fire_target(fighter, red)
            cooldown_ready = fighter[7] <= 0.0
            ready = cooldown_ready and has_target
            if ready:
                opportunities += 1
                opportunities_by_agent[row] = 1.0
            if not requested:
                continue

            requested_inputs += 1
            requested_by_agent[row] = 1.0
            if ready:
                attempts += 1
                attempts_by_agent[row] = 1.0
                applied[row, 2] = 1.0
            else:
                blocked_attempts += 1
                blocked_by_agent[row] = 1.0
                if not cooldown_ready:
                    cooldown_blocked_attempts += 1
                    cooldown_blocked_by_agent[row] = 1.0

        return applied, {
            "requested_fire_inputs": float(requested_inputs),
            "fire_attempts": float(attempts),
            "fire_opportunities": float(opportunities),
            "blocked_fire_attempts": float(blocked_attempts),
            "cooldown_blocked_fire_attempts": float(cooldown_blocked_attempts),
            "requested_fire_inputs_by_agent": requested_by_agent,
            "fire_attempts_by_agent": attempts_by_agent,
            "fire_opportunities_by_agent": opportunities_by_agent,
            "blocked_fire_attempts_by_agent": blocked_by_agent,
            "cooldown_blocked_fire_attempts_by_agent": cooldown_blocked_by_agent,
        }

    @staticmethod
    def _apply_learner_motion_floor(
        fighters: np.ndarray, learner_actions: np.ndarray
    ) -> np.ndarray:
        applied = np.asarray(learner_actions, dtype=np.float64).copy()
        blue = fighters[(fighters[:, 0] == TEAM_LEARN)]
        for row, fighter in enumerate(blue):
            if row >= len(applied) or fighter[6] <= 0.0:
                continue
            applied[row, 0] = max(float(applied[row, 0]), MIN_LEARNER_ACCELERATION)
        return applied

    @staticmethod
    def _fire_feedback(action_info: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
        bonus = 0.0
        attempts = int(action_info.get("fire_attempts", 0.0))
        opportunities = int(action_info.get("fire_opportunities", 0.0))
        blocked_attempts = int(action_info.get("blocked_fire_attempts", 0.0))
        cooldown_blocked_attempts = int(
            action_info.get("cooldown_blocked_fire_attempts", 0.0)
        )
        missed_opportunities = max(0, opportunities - attempts)
        targetless_blocked_attempts = max(
            0, blocked_attempts - cooldown_blocked_attempts
        )

        bonus += FIRE_SUCCESS_BONUS * attempts
        bonus -= MISSED_FIRE_OPPORTUNITY_PENALTY * missed_opportunities
        bonus -= BAD_FIRE_ATTEMPT_PENALTY * targetless_blocked_attempts
        bonus -= COOLDOWN_FIRE_ATTEMPT_PENALTY * cooldown_blocked_attempts

        public_info = {
            key: value
            for key, value in action_info.items()
            if not key.endswith("_by_agent")
        }
        return bonus, {
            **public_info,
            "fire_successes": float(attempts),
            "missed_fire_opportunities": float(missed_opportunities),
        }

    @staticmethod
    def _agent_fire_feedback(
        action_info: Dict[str, object], agent_count: int
    ) -> np.ndarray:
        def get_array(key: str) -> np.ndarray:
            values = action_info.get(key)
            if values is None:
                return np.zeros((agent_count,), dtype=np.float32)
            array = np.asarray(values, dtype=np.float32)
            if len(array) < agent_count:
                padded = np.zeros((agent_count,), dtype=np.float32)
                padded[: len(array)] = array
                return padded
            return array[:agent_count]

        attempts = get_array("fire_attempts_by_agent")
        opportunities = get_array("fire_opportunities_by_agent")
        blocked_attempts = get_array("blocked_fire_attempts_by_agent")
        cooldown_blocked_attempts = get_array("cooldown_blocked_fire_attempts_by_agent")
        missed_opportunities = np.maximum(0.0, opportunities - attempts)
        targetless_blocked_attempts = np.maximum(
            0.0, blocked_attempts - cooldown_blocked_attempts
        )

        return (
            FIRE_SUCCESS_BONUS * attempts
            - MISSED_FIRE_OPPORTUNITY_PENALTY * missed_opportunities
            - BAD_FIRE_ATTEMPT_PENALTY * targetless_blocked_attempts
            - COOLDOWN_FIRE_ATTEMPT_PENALTY * cooldown_blocked_attempts
        ).astype(np.float32)

    @staticmethod
    def _has_fire_target(fighter: np.ndarray, red: np.ndarray) -> bool:
        if len(red) == 0:
            return False

        for delta in red[:, 2:4] - fighter[2:4]:
            desired = math.atan2(float(delta[1]), float(delta[0]))
            if (
                abs(_angle_delta(desired, float(fighter[4])))
                <= MISSILE_SEEKER_HALF_ANGLE
            ):
                return True
        return False

    @staticmethod
    def _agent_engagement_bonus(
        prev_fighters: np.ndarray, next_fighters: np.ndarray
    ) -> np.ndarray:
        next_blue = next_fighters[next_fighters[:, 0] == TEAM_LEARN]
        rewards = np.zeros((len(next_blue),), dtype=np.float32)
        field_w = 1600.0
        field_h = 900.0
        diag = math.hypot(field_w, field_h)
        if diag <= 0.0:
            return rewards

        prev_blue = prev_fighters[prev_fighters[:, 0] == TEAM_LEARN]
        prev_red = prev_fighters[
            (prev_fighters[:, 0] == TEAM_RULE) & (prev_fighters[:, 6] > 0.0)
        ]
        next_red = next_fighters[
            (next_fighters[:, 0] == TEAM_RULE) & (next_fighters[:, 6] > 0.0)
        ]

        for row, fighter in enumerate(next_blue):
            if row >= len(prev_blue) or fighter[6] <= 0.0:
                continue
            prev_distance = ToyAcaiPPOEnv._nearest_enemy_distance(
                prev_blue[row], prev_red
            )
            next_distance = ToyAcaiPPOEnv._nearest_enemy_distance(fighter, next_red)
            if not math.isfinite(prev_distance) or not math.isfinite(next_distance):
                continue

            closing = (prev_distance - next_distance) / diag
            rewards[row] = ENGAGEMENT_CLOSING_BONUS_SCALE * closing
            if next_distance > diag * 0.35 and closing <= 0.0:
                rewards[row] -= DISTANT_LOITER_PENALTY
        return rewards

    @staticmethod
    def _nearest_enemy_distance(fighter: np.ndarray, enemies: np.ndarray) -> float:
        if len(enemies) == 0 or fighter[6] <= 0.0:
            return math.inf
        deltas = enemies[:, 2:4] - fighter[2:4]
        return math.sqrt(float(np.min(np.sum(deltas * deltas, axis=1))))

    @staticmethod
    def _agent_low_speed_penalty(fighters: np.ndarray) -> np.ndarray:
        blue = fighters[fighters[:, 0] == TEAM_LEARN]
        penalties = np.zeros((len(blue),), dtype=np.float32)
        speed_ratio = blue[:, 5] / MAX_SPEED
        alive = blue[:, 6] > 0.0
        penalties[alive] = (
            np.clip(0.25 - speed_ratio[alive], 0.0, 0.25)
            * LOW_SPEED_PENALTY_SCALE
        )
        return penalties

    @staticmethod
    def _agent_oob_penalty(fighters: np.ndarray) -> np.ndarray:
        blue = fighters[fighters[:, 0] == TEAM_LEARN]
        penalties = (
            np.clip(blue[:, 8] / OUT_OF_BOUNDS_DEATH_TIME, 0.0, 1.0) * 0.02
        ).astype(np.float32)
        penalties[blue[:, 6] <= 0.0] = 0.0
        return penalties

    @staticmethod
    def _turn_penalty(fighters: np.ndarray, learner_actions: np.ndarray) -> np.ndarray:
        blue = fighters[fighters[:, 0] == TEAM_LEARN]
        actions = np.asarray(learner_actions, dtype=np.float64)
        penalties = np.zeros((len(blue),), dtype=np.float32)
        count = min(len(blue), len(actions))
        if count == 0:
            return penalties
        alive = blue[:count, 6] > 0.0
        penalties[:count] = (
            np.square(np.clip(actions[:count, 1], -1.0, 1.0)) * TURN_PENALTY_SCALE
        ).astype(np.float32)
        penalties[np.where(~alive)[0]] = 0.0
        return penalties
