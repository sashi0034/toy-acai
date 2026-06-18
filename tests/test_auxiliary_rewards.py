import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

import toy_acai_rl.env as env_module  # noqa: E402
from toy_acai_rl.env import (  # noqa: E402
    AUX_DEATH_PENALTY,
    AUX_KILL_REWARD,
    AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD,
    AUX_LOW_MOVEMENT_PENALTY_PER_STEP,
    AUX_MISSILE_TRACKING_PENALTY_DISTANCE_SCALE,
    AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP,
    AUX_MISSILE_FIRE_REWARD,
    AUX_FIRE_INPUT_IN_RANGE_REWARD_PER_STEP,
    AUX_FIRE_INPUT_OUT_OF_RANGE_PENALTY_PER_STEP,
    AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP,
    AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP,
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
    blue_yaws=None,
    red_positions=None,
    red_yaws=None,
    blue_cooldowns=None,
    blue_out_of_bounds_time=None,
    missiles=None,
):
    fighters = np.zeros((8, 9), dtype=np.float64)
    fighters[:4, 0] = TEAM_LEARN
    fighters[:4, 1] = np.arange(4)
    fighters[:4, 6] = blue_health
    if blue_cooldowns is not None:
        fighters[:4, 7] = blue_cooldowns
    if blue_out_of_bounds_time is not None:
        fighters[:4, 8] = blue_out_of_bounds_time
    if blue_positions is not None:
        fighters[:4, 2:4] = np.asarray(blue_positions, dtype=np.float64)
    if blue_yaws is not None:
        fighters[:4, 4] = np.asarray(blue_yaws, dtype=np.float64)
    fighters[4:, 0] = TEAM_RULE
    fighters[4:, 1] = np.arange(4)
    fighters[4:, 6] = red_health
    if red_positions is not None:
        fighters[4:, 2:4] = np.asarray(red_positions, dtype=np.float64)
    if red_yaws is not None:
        fighters[4:, 4] = np.asarray(red_yaws, dtype=np.float64)
    if hit_events is None:
        hit_events = np.zeros((0, 4), dtype=np.float64)
    if missiles is None:
        missiles = np.zeros((0, 9), dtype=np.float64)
    return {
        "fighters": fighters,
        "missiles": np.asarray(missiles, dtype=np.float64),
        "hit_events": np.asarray(hit_events, dtype=np.float64),
        "battlefield": (0.0, 0.0, 1600.0, 900.0),
    }

def missile_row(
    *,
    x=100.0,
    y=0.0,
    yaw=0.0,
    speed=240.0,
    age=0.5,
    lock_lost_time=0.0,
    team=TEAM_RULE,
    target=0.0,
    missile_id=1.0,
):
    return [
        x,
        y,
        yaw,
        speed,
        age,
        lock_lost_time,
        float(team),
        float(target),
        float(missile_id),
    ]


class AuxiliaryRewardsTest(unittest.TestCase):
    def test_no_step_reward_goes_to_alive_blue_agents_by_default(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(blue_health=(1.0, 0.0, 1.0, 0.0))
        )

        np.testing.assert_allclose(
            rewards,
            np.zeros((4,), dtype=np.float32),
            atol=1e-8,
        )
        self.assertAlmostEqual(info["survival_reward"], 0.0)
        self.assertEqual(info["blue_kills"], 0.0)

    def test_out_of_bounds_penalty_applies_while_blue_agent_is_outside(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(
                blue_health=(1.0, 0.0, 1.0, 0.0),
                blue_out_of_bounds_time=(0.25, 0.0, 0.0, 0.0),
            )
        )

        np.testing.assert_allclose(
            rewards,
            np.array(
                [
                    -AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            atol=1e-8,
        )
        self.assertAlmostEqual(info["survival_reward"], 0.0)
        self.assertAlmostEqual(
            info["out_of_bounds_penalty"],
            AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP,
        )

    def test_learner_count_limits_rewarded_blue_agents(self):
        rewards, info = auxiliary_agent_rewards(make_obs(), learner_count=1)

        np.testing.assert_allclose(
            rewards,
            np.array([0.0], dtype=np.float32),
        )
        self.assertAlmostEqual(info["survival_reward"], 0.0)

    def test_opponent_count_limits_env_red_team_size(self):
        core = FakeCore(make_obs())
        ToyAcaiPPOEnv(core, max_steps=1, learner_count=1, opponent_count=2, rng=object())

        self.assertEqual(core.last_kwargs["active_blue_count"], 1)
        self.assertEqual(core.last_kwargs["active_red_count"], 2)

    def test_opponent_count_sets_terminal_score_denominator(self):
        next_obs = make_obs(
            blue_health=(1.0, 0.0, 0.0, 0.0),
            red_health=(0.0, 1.0, 0.0, 0.0),
        )
        core = FakeCore(next_obs)
        env = ToyAcaiPPOEnv(core, max_steps=1, learner_count=1, opponent_count=2, rng=object())
        env.last_obs = make_obs(blue_health=(1.0, 0.0, 0.0, 0.0))
        env.opponent = FakeOpponent()

        original_build_agent_observations = env_module.build_agent_observations
        env_module.build_agent_observations = fake_build_agent_observations
        try:
            result = env.step(np.zeros((1, 3), dtype=np.float64))
        finally:
            env_module.build_agent_observations = original_build_agent_observations

        score = terminal_score(
            blue_alive=1,
            red_alive=1,
            episode_steps=1,
            max_steps=1,
            team_size=1,
            opponent_team_size=2,
        )
        expected = np.array([score], dtype=np.float32)
        np.testing.assert_allclose(result.rewards, expected, atol=1e-6)
        self.assertAlmostEqual(result.info["terminal_score"], score)
        self.assertEqual(result.info["red_alive"], 1.0)

    def test_missile_tracking_penalty_applies_when_enemy_missile_points_at_blue(self):
        current_obs = make_obs(
            red_health=(0.0, 0.0, 0.0, 0.0),
            blue_positions=[
                [0.0, 0.0],
                [500.0, 300.0],
                [500.0, 400.0],
                [500.0, 500.0],
            ],
            missiles=[missile_row(x=200.0, y=0.0, yaw=np.pi)],
        )

        rewards, info = auxiliary_agent_rewards(current_obs)

        expected = np.zeros((4,), dtype=np.float32)
        expected[0] -= AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(
            info["missile_tracking_penalty"],
            AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP,
        )

    def test_missile_tracking_penalty_scales_with_distance_and_ignores_friendly(self):
        current_obs = make_obs(
            red_health=(0.0, 0.0, 0.0, 0.0),
            blue_positions=[
                [0.0, 0.0],
                [400.0, 400.0],
                [500.0, 400.0],
                [600.0, 400.0],
            ],
            missiles=[
                missile_row(x=400.0, y=0.0, yaw=np.pi),
                missile_row(x=200.0, y=0.0, yaw=np.pi, team=TEAM_LEARN),
            ],
        )

        rewards, info = auxiliary_agent_rewards(current_obs)

        expected_penalty = (
            AUX_MISSILE_TRACKING_PENALTY_MAX_PER_STEP
            * AUX_MISSILE_TRACKING_PENALTY_DISTANCE_SCALE
            / 400.0
        )
        expected = np.zeros((4,), dtype=np.float32)
        expected[0] -= expected_penalty
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["missile_tracking_penalty"], expected_penalty)

    def test_missile_tracking_penalty_requires_missile_to_face_blue(self):
        current_obs = make_obs(
            red_health=(0.0, 0.0, 0.0, 0.0),
            missiles=[missile_row(x=200.0, y=0.0, yaw=0.0)],
        )

        rewards, info = auxiliary_agent_rewards(current_obs)

        np.testing.assert_allclose(rewards, np.zeros((4,), dtype=np.float32), atol=1e-8)
        self.assertAlmostEqual(info["missile_tracking_penalty"], 0.0)

    def test_nearest_enemy_facing_reward_and_penalty_use_same_amount(self):
        current_obs = make_obs(
            blue_positions=[
                [0.0, 0.0],
                [0.0, 200.0],
                [0.0, 300.0],
                [0.0, 400.0],
            ],
            blue_yaws=(0.0, 0.0, 0.0, 0.0),
            red_positions=[
                [100.0, 0.0],
                [500.0, 0.0],
                [500.0, 100.0],
                [500.0, 200.0],
            ],
            red_yaws=(np.pi, 0.0, 0.0, 0.0),
        )

        rewards, info = auxiliary_agent_rewards(current_obs, learner_count=1)

        np.testing.assert_allclose(rewards, np.array([0.0], dtype=np.float32), atol=1e-8)
        self.assertAlmostEqual(
            info["nearest_enemy_facing_reward"],
            AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP,
        )
        self.assertAlmostEqual(
            info["nearest_enemy_facing_penalty"],
            AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP,
        )

    def test_nearest_enemy_facing_uses_closest_living_enemy_only(self):
        current_obs = make_obs(
            blue_positions=[
                [0.0, 0.0],
                [0.0, 200.0],
                [0.0, 300.0],
                [0.0, 400.0],
            ],
            blue_yaws=(0.0, 0.0, 0.0, 0.0),
            red_positions=[
                [0.0, 100.0],
                [100.0, 0.0],
                [500.0, 100.0],
                [500.0, 200.0],
            ],
            red_yaws=(0.0, 0.0, 0.0, 0.0),
        )

        rewards, info = auxiliary_agent_rewards(current_obs, learner_count=1)

        np.testing.assert_allclose(rewards, np.array([0.0], dtype=np.float32), atol=1e-8)
        self.assertAlmostEqual(info["nearest_enemy_facing_reward"], 0.0)
        self.assertAlmostEqual(info["nearest_enemy_facing_penalty"], 0.0)

    def test_low_movement_penalty_applies_to_alive_in_bounds_agents(self):
        current_obs = make_obs(
            red_health=(0.0, 0.0, 0.0, 0.0),
            blue_health=(1.0, 1.0, 0.0, 1.0),
            blue_out_of_bounds_time=(0.0, 0.0, 0.0, 0.25),
        )
        distances = np.array(
            [
                AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD,
                AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD + 1.0,
                0.0,
                10.0,
            ],
            dtype=np.float64,
        )

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            low_movement_distances_1s=distances,
        )

        expected = np.zeros((4,), dtype=np.float32)
        expected[0] -= AUX_LOW_MOVEMENT_PENALTY_PER_STEP
        expected[3] -= AUX_OUT_OF_BOUNDS_PENALTY_PER_STEP
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(
            info["low_movement_penalty"],
            AUX_LOW_MOVEMENT_PENALTY_PER_STEP,
        )
        self.assertAlmostEqual(
            info["mean_movement_distance_1s"],
            (
                AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD
                + AUX_LOW_MOVEMENT_DISTANCE_THRESHOLD
                + 1.0
            )
            / 2.0,
        )

    def test_missile_fire_reward_uses_cooldown_increase(self):
        blue_positions = [
            [0.0, 0.0],
            [0.0, 50.0],
            [0.0, 100.0],
            [0.0, 150.0],
        ]
        red_positions = [
            [100.0, 0.0],
            [100.0, 150.0],
            [100.0, 300.0],
            [100.0, 450.0],
        ]
        previous_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
            blue_cooldowns=(0.0, 1.5, 0.0, 0.0),
        )
        current_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
            blue_cooldowns=(3.5, 1.4, 0.0, 3.5),
        )

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            previous_obs=previous_obs,
            nearest_enemy_facing_reward_per_step=0.0,
        )

        expected = np.zeros((4,), dtype=np.float32)
        expected[0] += AUX_MISSILE_FIRE_REWARD
        expected[3] += AUX_MISSILE_FIRE_REWARD
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["missile_fire_reward"], 2 * AUX_MISSILE_FIRE_REWARD)

    def test_missile_fire_reward_requires_enemy_in_fire_arc(self):
        previous_obs = make_obs(
            blue_positions=[[0.0, 0.0]] * 4,
            red_positions=[
                [-100.0, 0.0],
                [-100.0, 50.0],
                [-100.0, 100.0],
                [-100.0, 150.0],
            ],
            blue_cooldowns=(0.0, 0.0, 0.0, 0.0),
        )
        current_obs = make_obs(
            blue_positions=[[0.0, 0.0]] * 4,
            red_positions=[
                [-100.0, 0.0],
                [-100.0, 50.0],
                [-100.0, 100.0],
                [-100.0, 150.0],
            ],
            blue_cooldowns=(3.5, 0.0, 0.0, 0.0),
        )

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            previous_obs=previous_obs,
            nearest_enemy_facing_reward_per_step=0.0,
        )

        np.testing.assert_allclose(rewards, np.zeros((4,), dtype=np.float32), atol=1e-8)
        self.assertAlmostEqual(info["missile_fire_reward"], 0.0)

    def test_fire_input_in_range_reward_while_pressing_fire(self):
        blue_positions = [[0.0, 0.0]] * 4
        red_positions = [[100.0, 0.0], [100.0, 150.0], [100.0, 300.0], [100.0, 450.0]]
        previous_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
        )
        current_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
        )
        learner_actions = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            previous_obs=previous_obs,
            learner_actions=learner_actions,
            learner_count=1,
        )

        expected = np.array(
            [
                AUX_FIRE_INPUT_IN_RANGE_REWARD_PER_STEP
                + AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(
            info["fire_input_in_range_reward"],
            AUX_FIRE_INPUT_IN_RANGE_REWARD_PER_STEP,
        )
        self.assertAlmostEqual(info["fire_input_out_of_range_penalty"], 0.0)

    def test_fire_input_out_of_range_penalty_while_pressing_fire(self):
        previous_obs = make_obs(
            blue_positions=[[0.0, 0.0]] * 4,
            red_positions=[
                [-100.0, 0.0],
                [-100.0, 50.0],
                [-100.0, 100.0],
                [-100.0, 150.0],
            ],
            red_yaws=[np.pi, np.pi, np.pi, np.pi],
        )
        current_obs = make_obs(
            blue_positions=[[0.0, 0.0]] * 4,
            red_positions=[
                [-100.0, 0.0],
                [-100.0, 50.0],
                [-100.0, 100.0],
                [-100.0, 150.0],
            ],
            red_yaws=[np.pi, np.pi, np.pi, np.pi],
        )
        learner_actions = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            previous_obs=previous_obs,
            learner_actions=learner_actions,
            learner_count=1,
        )

        expected = np.array(
            [-AUX_FIRE_INPUT_OUT_OF_RANGE_PENALTY_PER_STEP],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["fire_input_in_range_reward"], 0.0)
        self.assertAlmostEqual(
            info["fire_input_out_of_range_penalty"],
            AUX_FIRE_INPUT_OUT_OF_RANGE_PENALTY_PER_STEP,
        )

    def test_fire_input_reward_ignored_without_fire_input(self):
        blue_positions = [[0.0, 0.0]] * 4
        red_positions = [[100.0, 0.0], [100.0, 150.0], [100.0, 300.0], [100.0, 450.0]]
        previous_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
        )
        current_obs = make_obs(
            blue_positions=blue_positions,
            red_positions=red_positions,
        )
        learner_actions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        rewards, info = auxiliary_agent_rewards(
            current_obs,
            previous_obs=previous_obs,
            learner_actions=learner_actions,
            learner_count=1,
        )

        self.assertAlmostEqual(info["fire_input_in_range_reward"], 0.0)
        self.assertAlmostEqual(info["fire_input_out_of_range_penalty"], 0.0)
        self.assertAlmostEqual(
            rewards[0],
            AUX_NEAREST_ENEMY_FACING_REWARD_PER_STEP,
        )

    def test_blue_hit_event_rewards_shooter_only(self):
        rewards, info = auxiliary_agent_rewards(make_obs(hit_events=[[2, 0, 4, 1]]))

        expected = np.zeros((4,), dtype=np.float32)
        expected[2] += AUX_KILL_REWARD
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["kill_reward"], AUX_KILL_REWARD)
        self.assertNotIn("team_kill_reward", info)
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

        expected = np.zeros((4,), dtype=np.float32)
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["kill_reward"], 0.0)
        self.assertEqual(info["blue_kills"], 0.0)
        self.assertEqual(info["hit_events"], 3.0)

    def test_blue_death_penalizes_lost_fighter_and_team(self):
        rewards, info = auxiliary_agent_rewards(
            make_obs(blue_health=(1.0, 0.0, 1.0, 1.0)),
            previous_obs=make_obs(),
        )

        expected = np.array(
            [
                0.0,
                -AUX_DEATH_PENALTY,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected, atol=1e-8)
        self.assertAlmostEqual(info["death_penalty"], AUX_DEATH_PENALTY)
        self.assertNotIn("team_loss_penalty", info)
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
            score,
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
        self.last_kwargs = None

    def BattlefieldEnv(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeBattlefieldEnv(self.next_obs)


class FakeBattlefieldEnv:
    def __init__(self, next_obs):
        self.next_obs = next_obs

    def step(self, _actions):
        return self.next_obs


class FakeOpponent:
    def actions(self, _obs, fighter_count):
        return np.zeros((fighter_count, 3), dtype=np.float64)


def fake_build_agent_observations(_obs, **_kwargs):
    return np.zeros((4, 1), dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
