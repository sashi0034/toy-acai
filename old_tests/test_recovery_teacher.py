import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from toy_acai_rl.env import (  # noqa: E402
    RECOVERY_ROLLBACK_STEPS,
    RECOVERY_TEACHER_STEPS,
    TEAM_LEARN,
    TEAM_RULE,
    ToyAcaiPPOEnv,
)


def make_obs(blue_alive=True, hit_events=None):
    fighters = np.zeros((8, 9), dtype=np.float64)
    fighters[:4, 0] = TEAM_LEARN
    fighters[:4, 1] = np.arange(4)
    fighters[:4, 2] = 200.0
    fighters[:4, 3] = 450.0
    fighters[:4, 5] = 120.0
    fighters[:4, 6] = 1.0
    fighters[4:, 0] = TEAM_RULE
    fighters[4:, 1] = np.arange(4)
    fighters[4:, 2] = 1000.0
    fighters[4:, 3] = 450.0
    fighters[4:, 4] = np.pi
    fighters[4:, 5] = 120.0
    fighters[4:, 6] = 1.0
    if not blue_alive:
        fighters[0, 6] = 0.0
    if hit_events is None:
        hit_events = np.zeros((0, 4), dtype=np.float64)
    return {
        "fighters": fighters,
        "missiles": np.zeros((0, 9), dtype=np.float64),
        "hit_events": np.asarray(hit_events, dtype=np.float64),
        "battlefield": (0.0, 0.0, 1600.0, 900.0),
    }


def copy_obs(obs):
    return {
        "fighters": np.asarray(obs["fighters"], dtype=np.float64).copy(),
        "missiles": np.asarray(obs["missiles"], dtype=np.float64).copy(),
        "hit_events": np.asarray(obs["hit_events"], dtype=np.float64).copy(),
        "battlefield": tuple(obs["battlefield"]),
    }


class RecoveryTeacherTest(unittest.TestCase):
    def test_missile_death_generates_first_segment_teacher_examples(self):
        core = FakeCore()
        env = ToyAcaiPPOEnv(
            core,
            max_steps=200,
            learner_count=1,
            opponent_count=1,
            rng=object(),
            random_start_positions=False,
        )
        env.opponent = FakeOpponent()
        env.reset()

        result = None
        for _ in range(RECOVERY_ROLLBACK_STEPS):
            result = env.step(np.zeros((1, 3), dtype=np.float64))

        self.assertIsNotNone(result)
        self.assertEqual(result.info["recovery_teacher_attempts"], 1.0)
        self.assertEqual(result.info["recovery_teacher_successes"], 1.0)
        self.assertEqual(result.info["recovery_teacher_examples"], float(RECOVERY_TEACHER_STEPS))
        self.assertEqual(result.info["recovery_teacher_candidates"], 256.0)

        examples = env.pop_recovery_teacher_examples()
        self.assertEqual(len(examples), RECOVERY_TEACHER_STEPS)
        for example in examples:
            self.assertEqual(example.agent_id, 0)
            np.testing.assert_allclose(
                example.action,
                np.array([1.0, -1.0, 0.0], dtype=np.float32),
            )
        self.assertEqual(core.envs[0].restore_calls, 0)
        self.assertGreater(core.envs[1].restore_calls, 0)


class FakeCore:
    FIGHTER_COUNT = 8
    TEAM_FIGHTER_COUNT = 4

    def __init__(self):
        self.envs = []

    def BattlefieldEnv(self, **_kwargs):
        role = "main" if not self.envs else "recovery"
        env = FakeBattlefieldEnv(role)
        self.envs.append(env)
        return env


class FakeBattlefieldEnv:
    def __init__(self, role):
        self.role = role
        self.obs = make_obs()
        self.step_calls = 0
        self.restore_calls = 0

    def reset(self):
        self.obs = make_obs()
        self.step_calls = 0
        return copy_obs(self.obs)

    def snapshot(self):
        return copy_obs(self.obs)

    def restore_snapshot(self, snapshot):
        self.restore_calls += 1
        self.obs = copy_obs(snapshot)
        return copy_obs(self.obs)

    def step(self, actions):
        self.step_calls += 1
        if self.role == "main":
            if self.step_calls >= RECOVERY_ROLLBACK_STEPS:
                self.obs = make_obs(
                    blue_alive=False,
                    hit_events=[[4, TEAM_RULE, 0, TEAM_LEARN]],
                )
            else:
                self.obs = make_obs()
            return copy_obs(self.obs)

        desired = np.array([1.0, -1.0, 0.0], dtype=np.float64)
        self.obs = copy_obs(self.obs)
        self.obs["hit_events"] = np.zeros((0, 4), dtype=np.float64)
        if not np.allclose(actions[0], desired):
            self.obs = make_obs(
                blue_alive=False,
                hit_events=[[4, TEAM_RULE, 0, TEAM_LEARN]],
            )
        return copy_obs(self.obs)


class FakeOpponent:
    def actions(self, _obs, fighter_count):
        return np.zeros((fighter_count, 3), dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
