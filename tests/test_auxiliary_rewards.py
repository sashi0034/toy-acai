import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

import toy_acai_rl.env as env_module  # noqa: E402
from toy_acai_rl.env import (  # noqa: E402
    AUX_KILL_REWARD,
    AUX_SURVIVAL_REWARD_PER_STEP,
    TEAM_LEARN,
    TEAM_RULE,
    ToyAcaiPPOEnv,
    auxiliary_agent_rewards,
    terminal_score,
)


def make_obs(blue_health=(1.0, 1.0, 1.0, 1.0), red_health=(1.0, 1.0, 1.0, 1.0), hit_events=None):
    fighters = np.zeros((8, 9), dtype=np.float64)
    fighters[:4, 0] = TEAM_LEARN
    fighters[:4, 1] = np.arange(4)
    fighters[:4, 6] = blue_health
    fighters[4:, 0] = TEAM_RULE
    fighters[4:, 1] = np.arange(4)
    fighters[4:, 6] = red_health
    if hit_events is None:
        hit_events = np.zeros((0, 4), dtype=np.float64)
    return {
        "fighters": fighters,
        "missiles": np.zeros((0, 8), dtype=np.float64),
        "hit_events": np.asarray(hit_events, dtype=np.float64),
        "battlefield": (0.0, 0.0, 1600.0, 900.0),
    }


class AuxiliaryRewardsTest(unittest.TestCase):
    def test_survival_reward_only_goes_to_alive_blue_agents(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(blue_health=(1.0, 0.0, 1.0, 0.0))
        )

        np.testing.assert_allclose(
            rewards,
            np.array([0.001, 0.0, 0.001, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(info["survival_reward"], 0.002)
        self.assertEqual(info["blue_kills"], 0.0)

    def test_blue_hit_event_rewards_only_the_shooter(self):
        rewards, info = auxiliary_agent_rewards(make_obs(hit_events=[[2, 0, 4, 1]]))

        expected = np.full((4,), AUX_SURVIVAL_REWARD_PER_STEP, dtype=np.float32)
        expected[2] += AUX_KILL_REWARD
        np.testing.assert_allclose(rewards, expected)
        self.assertAlmostEqual(info["kill_reward"], AUX_KILL_REWARD)
        self.assertEqual(info["blue_kills"], 1.0)
        self.assertEqual(info["hit_events"], 1.0)

    def test_non_blue_or_non_red_hit_events_do_not_reward_blue(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(
                hit_events=[
                    [4, 1, 0, 0],
                    [4, 0, 5, 1],
                    [1, 0, 2, 0],
                ]
            )
        )

        expected = np.full((4,), AUX_SURVIVAL_REWARD_PER_STEP, dtype=np.float32)
        np.testing.assert_allclose(rewards, expected)
        self.assertAlmostEqual(info["kill_reward"], 0.0)
        self.assertEqual(info["blue_kills"], 0.0)
        self.assertEqual(info["hit_events"], 3.0)

    def test_terminal_step_combines_terminal_score_and_auxiliary_rewards(self):
        next_obs = make_obs(red_health=(0.0, 1.0, 1.0, 1.0), hit_events=[[1, 0, 4, 1]])
        core = FakeCore(next_obs)
        env = ToyAcaiPPOEnv(core, max_steps=1, rng=object())
        env.last_obs = make_obs()
        env.opponent = FakeOpponent()

        original_build_agent_observations = env_module.build_agent_observations
        env_module.build_agent_observations = fake_build_agent_observations
        try:
            result = env.step(np.zeros((4, 3), dtype=np.float64))
        finally:
            env_module.build_agent_observations = original_build_agent_observations

        score = terminal_score(
            blue_alive=4,
            red_alive=3,
            episode_steps=1,
            max_steps=1,
            team_size=4,
        )
        expected = np.full((4,), score + AUX_SURVIVAL_REWARD_PER_STEP, dtype=np.float32)
        expected[1] += AUX_KILL_REWARD
        np.testing.assert_allclose(result.rewards, expected)
        self.assertTrue(result.done)
        self.assertAlmostEqual(result.info["terminal_score"], score)
        self.assertEqual(result.info["blue_kills"], 1.0)


class FakeCore:
    FIGHTER_COUNT = 8
    TEAM_FIGHTER_COUNT = 4

    def __init__(self, next_obs):
        self.next_obs = next_obs

    def BattlefieldEnv(self, **_kwargs):
        return FakeBattlefieldEnv(self.next_obs)


class FakeBattlefieldEnv:
    def __init__(self, next_obs):
        self.next_obs = next_obs

    def step(self, _actions):
        return self.next_obs


class FakeOpponent:
    def actions(self, _obs, fighter_count):
        return np.zeros((fighter_count, 3), dtype=np.float64)


def fake_build_agent_observations(_obs):
    return np.zeros((4, 1), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
