import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from toy_acai_rl.env import (  # noqa: E402
    RANDOM_START_X_RANGE,
    RANDOM_START_Y_RANGE,
    RANDOM_START_YAW_JITTER,
    TEAM_LEARN,
    TEAM_RULE,
    ToyAcaiPPOEnv,
)


FIELD_W = 1600.0
FIELD_H = 900.0


def make_obs():
    fighters = np.zeros((8, 9), dtype=np.float64)
    fighters[:4, 0] = TEAM_LEARN
    fighters[:4, 1] = np.arange(4)
    fighters[:4, 2] = 160.0
    fighters[:4, 3] = 180.0 + np.arange(4) * 90.0
    fighters[:4, 4] = 0.0
    fighters[:4, 5] = 126.0
    fighters[:4, 6] = 1.0
    fighters[4:, 0] = TEAM_RULE
    fighters[4:, 1] = np.arange(4)
    fighters[4:, 2] = 1440.0
    fighters[4:, 3] = 180.0 + np.arange(4) * 90.0
    fighters[4:, 4] = math.pi
    fighters[4:, 5] = 126.0
    fighters[4:, 6] = 1.0
    return {
        "fighters": fighters,
        "missiles": np.zeros((0, 9), dtype=np.float64),
        "hit_events": np.zeros((0, 4), dtype=np.float64),
        "battlefield": (0.0, 0.0, FIELD_W, FIELD_H),
    }


class RandomStartPositionsTest(unittest.TestCase):
    def test_reset_randomizes_active_fighters_within_start_bounds(self):
        initial_obs = make_obs()
        core = FakeCore(FakeBattlefieldEnv(initial_obs))
        env = ToyAcaiPPOEnv(
            core,
            max_steps=1,
            random_start_positions=True,
            learner_count=2,
            opponent_count=3,
            rng=np.random.RandomState(7),
        )

        env.reset()

        poses = core.env.last_poses
        self.assertIsNotNone(poses)
        self.assertEqual(poses.shape, (8, 3))
        self.assert_start_bounds(poses[:2], TEAM_LEARN)
        self.assert_start_bounds(poses[4:7], TEAM_RULE)
        np.testing.assert_allclose(poses[2:4], initial_obs["fighters"][2:4, 2:5])
        np.testing.assert_allclose(poses[7:8], initial_obs["fighters"][7:8, 2:5])

    def test_random_start_positions_can_be_disabled_without_binding(self):
        core = FakeCore(FakeBattlefieldEnvWithoutPose(make_obs()))
        env = ToyAcaiPPOEnv(
            core,
            max_steps=1,
            random_start_positions=False,
            learner_count=1,
            opponent_count=1,
            rng=np.random.RandomState(1),
        )

        observations = env.reset()

        self.assertEqual(observations.shape[0], 1)

    def assert_start_bounds(self, poses, team_id):
        x_low, x_high = RANDOM_START_X_RANGE
        y_low, y_high = RANDOM_START_Y_RANGE
        self.assertTrue(np.all(poses[:, 0] >= x_low * FIELD_W))
        self.assertTrue(np.all(poses[:, 0] <= x_high * FIELD_W))
        self.assertTrue(np.all(poses[:, 1] >= y_low * FIELD_H))
        self.assertTrue(np.all(poses[:, 1] <= y_high * FIELD_H))
        base_yaw = 0.0 if team_id == TEAM_LEARN else math.pi
        self.assertTrue(np.all(poses[:, 2] >= base_yaw - RANDOM_START_YAW_JITTER))
        self.assertTrue(np.all(poses[:, 2] <= base_yaw + RANDOM_START_YAW_JITTER))


class FakeCore:
    FIGHTER_COUNT = 8
    TEAM_FIGHTER_COUNT = 4

    def __init__(self, env):
        self.env = env

    def BattlefieldEnv(self, **_kwargs):
        return self.env


class FakeBattlefieldEnv:
    def __init__(self, obs):
        self.obs = copy_obs(obs)
        self.last_poses = None

    def reset(self):
        return copy_obs(self.obs)

    def set_fighter_poses(self, poses):
        self.last_poses = np.asarray(poses, dtype=np.float64).copy()
        next_obs = copy_obs(self.obs)
        next_obs["fighters"][:, 2:5] = self.last_poses
        self.obs = next_obs
        return copy_obs(next_obs)

    def step(self, _actions):
        return copy_obs(self.obs)


class FakeBattlefieldEnvWithoutPose:
    def __init__(self, obs):
        self.obs = copy_obs(obs)

    def reset(self):
        return copy_obs(self.obs)

    def step(self, _actions):
        return copy_obs(self.obs)


def copy_obs(obs):
    return {
        "fighters": np.asarray(obs["fighters"], dtype=np.float64).copy(),
        "missiles": np.asarray(obs["missiles"], dtype=np.float64).copy(),
        "hit_events": np.asarray(obs["hit_events"], dtype=np.float64).copy(),
        "battlefield": tuple(obs["battlefield"]),
    }


if __name__ == "__main__":
    unittest.main()
