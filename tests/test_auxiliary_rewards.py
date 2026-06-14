import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

import toy_acai_rl.env as env_module  # noqa: E402
from toy_acai_rl.env import (  # noqa: E402
    AUX_ALIVE_ADVANTAGE_REWARD_PER_STEP,
    AUX_DEATH_PENALTY,
    AUX_KILL_REWARD,
    AUX_MOVEMENT_REWARD_PER_DISTANCE,
    AUX_SURVIVAL_REWARD_PER_STEP,
    AUX_TEAM_KILL_REWARD,
    AUX_TEAM_LOSS_PENALTY,
    TEAM_LEARN,
    TEAM_RULE,
    ToyAcaiPPOEnv,
    auxiliary_agent_rewards,
    terminal_score,
)


def make_obs(
    blue_health=(1.0, 1.0, 1.0, 1.0),
    red_health=(1.0, 1.0, 1.0, 1.0),
    hit_events=None,
    blue_positions=None,
):
    fighters = np.zeros((8, 9), dtype=np.float64)
    fighters[:4, 0] = TEAM_LEARN
    fighters[:4, 1] = np.arange(4)
    fighters[:4, 6] = blue_health
    if blue_positions is not None:
        fighters[:4, 2:4] = np.asarray(blue_positions, dtype=np.float64)
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


def base_step_reward(blue_alive=4, red_alive=4):
    return (
        AUX_SURVIVAL_REWARD_PER_STEP
        + AUX_ALIVE_ADVANTAGE_REWARD_PER_STEP * ((blue_alive - red_alive) / 4.0)
    )


class AuxiliaryRewardsTest(unittest.TestCase):
    def test_survival_reward_only_goes_to_alive_blue_agents(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(blue_health=(1.0, 0.0, 1.0, 0.0))
        )

        alive_reward = base_step_reward(blue_alive=2, red_alive=4)
        np.testing.assert_allclose(
            rewards,
            np.array([alive_reward, 0.0, alive_reward, 0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(info["survival_reward"], 2 * AUX_SURVIVAL_REWARD_PER_STEP)
        self.assertAlmostEqual(info["advantage_reward"], 2 * (alive_reward - AUX_SURVIVAL_REWARD_PER_STEP))
        self.assertEqual(info["blue_kills"], 0.0)

    def test_movement_distance_adds_tiny_reward_to_alive_agents(self):
        previous_obs = make_obs(
            blue_health=(1.0, 1.0, 0.0, 1.0),
            blue_positions=[
                [0.0, 0.0],
                [100.0, 0.0],
                [200.0, 0.0],
                [300.0, 0.0],
            ]
        )
        current_obs = make_obs(
            blue_health=(1.0, 1.0, 0.0, 1.0),
            blue_positions=[
                [3.0, 4.0],
                [100.0, 0.0],
                [212.0, 0.0],
                [300.0, 6.0],
            ],
        )

        rewards, info = auxiliary_agent_rewards(current_obs, previous_obs=previous_obs)

        reward_distance = np.hypot(1600.0, 900.0) * 0.25
        expected = np.array(
            [
                base_step_reward(blue_alive=3, red_alive=4)
                + (5.0 / reward_distance) * AUX_MOVEMENT_REWARD_PER_DISTANCE,
                base_step_reward(blue_alive=3, red_alive=4),
                0.0,
                base_step_reward(blue_alive=3, red_alive=4)
                + (6.0 / reward_distance) * AUX_MOVEMENT_REWARD_PER_DISTANCE,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(
            info["movement_reward"],
            ((5.0 + 6.0) / reward_distance) * AUX_MOVEMENT_REWARD_PER_DISTANCE,
        )
        self.assertAlmostEqual(info["mean_movement_distance"], (5.0 + 0.0 + 6.0) / 3.0)

    def test_blue_hit_event_rewards_shooter_and_surviving_team(self):
        rewards, info = auxiliary_agent_rewards(make_obs(hit_events=[[2, 0, 4, 1]]))

        expected = np.full((4,), base_step_reward(), dtype=np.float32)
        expected += AUX_TEAM_KILL_REWARD
        expected[2] += AUX_KILL_REWARD
        np.testing.assert_allclose(rewards, expected)
        self.assertAlmostEqual(info["kill_reward"], AUX_KILL_REWARD)
        self.assertAlmostEqual(info["team_kill_reward"], 4 * AUX_TEAM_KILL_REWARD)
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

        expected = np.full((4,), base_step_reward(), dtype=np.float32)
        np.testing.assert_allclose(rewards, expected)
        self.assertAlmostEqual(info["kill_reward"], 0.0)
        self.assertEqual(info["blue_kills"], 0.0)
        self.assertEqual(info["hit_events"], 3.0)

    def test_blue_death_penalizes_lost_fighter_and_team(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(blue_health=(1.0, 0.0, 1.0, 1.0)),
            previous_obs=make_obs(),
        )

        alive_reward = base_step_reward(blue_alive=3, red_alive=4)
        expected = np.array(
            [
                alive_reward - AUX_TEAM_LOSS_PENALTY,
                -AUX_DEATH_PENALTY - AUX_TEAM_LOSS_PENALTY,
                alive_reward - AUX_TEAM_LOSS_PENALTY,
                alive_reward - AUX_TEAM_LOSS_PENALTY,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected)
        self.assertAlmostEqual(info["death_penalty"], AUX_DEATH_PENALTY)
        self.assertAlmostEqual(info["team_loss_penalty"], 4 * AUX_TEAM_LOSS_PENALTY)
        self.assertEqual(info["blue_losses"], 1.0)

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
        expected = np.full(
            (4,),
            score + base_step_reward(blue_alive=4, red_alive=3) + AUX_TEAM_KILL_REWARD,
            dtype=np.float32,
        )
        expected[1] += AUX_KILL_REWARD
        np.testing.assert_allclose(result.rewards, expected, atol=1e-6)
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
