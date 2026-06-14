import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from toy_acai_rl.env import (  # noqa: E402
    MAX_SPEED,
    MAX_TRACKED_MISSILES,
    MISSILE_OBS_FEATURES,
    TEAM_LEARN,
    TEAM_RULE,
    build_agent_observations,
)


FIELD_W = 100.0
FIELD_H = 100.0
DIAG = math.hypot(FIELD_W, FIELD_H)
SELF_FEATURES = 13
OTHER_FEATURES = 13
OTHER_COUNT = 7
MISSILE_START = SELF_FEATURES + OTHER_COUNT * OTHER_FEATURES
EXPECTED_OBS_DIM = MISSILE_START + MAX_TRACKED_MISSILES * MISSILE_OBS_FEATURES


def make_obs(fighters=None, missiles=None):
    if fighters is None:
        fighters = np.zeros((8, 9), dtype=np.float64)
        fighters[:4, 0] = TEAM_LEARN
        fighters[:4, 1] = np.arange(4)
        fighters[:4, 2] = 20.0
        fighters[:4, 3] = 20.0 + np.arange(4) * 10.0
        fighters[:4, 4] = 0.0
        fighters[:4, 5] = MAX_SPEED * 0.25
        fighters[:4, 6] = 1.0
        fighters[4:, 0] = TEAM_RULE
        fighters[4:, 1] = np.arange(4)
        fighters[4:, 2] = 80.0
        fighters[4:, 3] = 20.0 + np.arange(4) * 10.0
        fighters[4:, 4] = math.pi
        fighters[4:, 5] = MAX_SPEED * 0.25
        fighters[4:, 6] = 1.0
    if missiles is None:
        missiles = np.zeros((0, 8), dtype=np.float64)
    return {
        "fighters": np.asarray(fighters, dtype=np.float64),
        "missiles": np.asarray(missiles, dtype=np.float64),
        "hit_events": np.zeros((0, 4), dtype=np.float64),
        "battlefield": (0.0, 0.0, FIELD_W, FIELD_H),
    }


class ObservationFeaturesTest(unittest.TestCase):
    def test_boundary_rays_are_relative_to_ownship_heading(self):
        fighters = np.zeros((8, 9), dtype=np.float64)
        fighters[:4, 0] = TEAM_LEARN
        fighters[4:, 0] = TEAM_RULE
        fighters[:, 6] = 1.0
        fighters[0, 2:5] = [50.0, 50.0, 0.0]

        obs = build_agent_observations(make_obs(fighters=fighters))[0]

        self.assertEqual(obs.shape[0], EXPECTED_OBS_DIM)
        self.assertAlmostEqual(obs[0], 50.0 / DIAG, places=6)
        self.assertAlmostEqual(obs[1], math.hypot(50.0, 50.0) / DIAG, places=6)
        self.assertAlmostEqual(obs[2], 50.0 / DIAG, places=6)
        self.assertAlmostEqual(obs[4], 50.0 / DIAG, places=6)

    def test_front_ray_gets_short_near_wall_and_zero_when_outside_facing_out(self):
        fighters = np.zeros((8, 9), dtype=np.float64)
        fighters[:4, 0] = TEAM_LEARN
        fighters[4:, 0] = TEAM_RULE
        fighters[:, 6] = 1.0
        fighters[0, 2:5] = [98.0, 50.0, 0.0]

        near_wall = build_agent_observations(make_obs(fighters=fighters))[0]
        self.assertAlmostEqual(near_wall[0], 2.0 / DIAG, places=6)

        fighters[0, 2:5] = [-5.0, 50.0, math.pi]
        outside_facing_out = build_agent_observations(make_obs(fighters=fighters))[0]
        self.assertAlmostEqual(outside_facing_out[0], 0.0, places=6)

    def test_other_fighter_features_use_ownship_forward_and_right_axes(self):
        fighters = np.zeros((8, 9), dtype=np.float64)
        fighters[:4, 0] = TEAM_LEARN
        fighters[4:, 0] = TEAM_RULE
        fighters[:, 6] = 1.0
        fighters[0, 2:5] = [50.0, 50.0, math.pi / 2.0]
        fighters[4, 2:5] = [60.0, 70.0, math.pi / 2.0]
        fighters[4, 5] = MAX_SPEED * 0.5
        fighters[4, 7] = 0.0

        obs = build_agent_observations(make_obs(fighters=fighters))[0]
        first_other = obs[SELF_FEATURES : SELF_FEATURES + OTHER_FEATURES]

        self.assertAlmostEqual(first_other[0], 20.0 / DIAG, places=6)
        self.assertAlmostEqual(first_other[1], -10.0 / DIAG, places=6)
        self.assertAlmostEqual(first_other[2], math.hypot(10.0, 20.0) / DIAG, places=6)
        self.assertAlmostEqual(first_other[3], 20.0 / math.hypot(10.0, 20.0), places=6)
        self.assertAlmostEqual(first_other[4], -10.0 / math.hypot(10.0, 20.0), places=6)
        self.assertEqual(first_other[9], -1.0)
        self.assertEqual(first_other[11], 1.0)
        self.assertEqual(first_other[12], 1.0)

    def test_missile_features_use_ownship_forward_and_right_axes(self):
        fighters = np.zeros((8, 9), dtype=np.float64)
        fighters[:4, 0] = TEAM_LEARN
        fighters[4:, 0] = TEAM_RULE
        fighters[:, 6] = 1.0
        fighters[0, 2:5] = [50.0, 50.0, math.pi / 2.0]

        missiles = np.array(
            [[60.0, 70.0, -math.pi / 2.0, MAX_SPEED * 0.5, 1.5, 0.2, TEAM_RULE, 0]],
            dtype=np.float64,
        )
        obs = build_agent_observations(make_obs(fighters=fighters, missiles=missiles))[0]
        missile = obs[MISSILE_START : MISSILE_START + MISSILE_OBS_FEATURES]

        distance = math.hypot(10.0, 20.0)
        self.assertAlmostEqual(missile[0], 20.0 / DIAG, places=6)
        self.assertAlmostEqual(missile[1], -10.0 / DIAG, places=6)
        self.assertAlmostEqual(missile[2], distance / DIAG, places=6)
        self.assertGreater(missile[3], 0.0)
        self.assertAlmostEqual(missile[4], 20.0 / distance, places=6)
        self.assertAlmostEqual(missile[5], -10.0 / distance, places=6)
        self.assertEqual(missile[11], -1.0)
        self.assertEqual(missile[12], 1.0)
        self.assertGreater(missile[13], 0.0)


if __name__ == "__main__":
    unittest.main()
