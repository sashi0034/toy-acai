import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import train_ppo  # noqa: E402
    from toy_acai_rl.env import StepResult  # noqa: E402


@unittest.skipUnless(HAS_TORCH, "torch is required")
class EpisodeInfoAggregatorTest(unittest.TestCase):
    def test_cumulative_keys_are_summed_across_steps(self):
        aggregator = train_ppo.EpisodeInfoAggregator()
        aggregator.add({"missile_tracking_penalty": 1.0, "kill_reward": 10.0})
        aggregator.add({"missile_tracking_penalty": 1.0})
        aggregator.add({"missile_fire_reward": 0.5})

        merged = aggregator.apply({"terminal_score": -2.0})

        self.assertAlmostEqual(merged["missile_tracking_penalty"], 2.0)
        self.assertAlmostEqual(merged["kill_reward"], 10.0)
        self.assertAlmostEqual(merged["missile_fire_reward"], 0.5)
        self.assertAlmostEqual(merged["death_penalty"], 0.0)
        self.assertAlmostEqual(merged["terminal_score"], -2.0)

    def test_apply_overrides_last_step_cumulative_keys(self):
        aggregator = train_ppo.EpisodeInfoAggregator()
        aggregator.add({"missile_tracking_penalty": 1.0})
        aggregator.add({"missile_tracking_penalty": 0.0, "death_penalty": 20.0})

        merged = aggregator.apply(
            {"missile_tracking_penalty": 0.0, "death_penalty": 20.0}
        )

        # 最終 step だけが残ると missile_tracking_penalty は 0 に見えてしまうが、
        # アグリゲータがエピソード合計で上書きする。
        self.assertAlmostEqual(merged["missile_tracking_penalty"], 1.0)
        self.assertAlmostEqual(merged["death_penalty"], 20.0)

    def test_mean_movement_distance_1s_is_averaged_over_steps(self):
        aggregator = train_ppo.EpisodeInfoAggregator()
        aggregator.add({"mean_movement_distance_1s": 2.0})
        aggregator.add({"mean_movement_distance_1s": 4.0})
        aggregator.add({"mean_movement_distance_1s": 0.0})

        merged = aggregator.apply({})

        self.assertAlmostEqual(merged["mean_movement_distance_1s"], (2.0 + 4.0 + 0.0) / 3.0)

    def test_mean_movement_distance_1s_zero_when_no_steps_recorded(self):
        aggregator = train_ppo.EpisodeInfoAggregator()

        merged = aggregator.apply({"mean_movement_distance_1s": 1.5})

        self.assertAlmostEqual(merged["mean_movement_distance_1s"], 0.0)


@unittest.skipUnless(HAS_TORCH, "torch is required")
class RunEpisodeValueGifTest(unittest.TestCase):
    def test_value_gif_receives_cumulative_per_agent_rewards(self):
        # GIF オーバーレイの reward 表示はフレーム時点までの累計報酬になる。
        # 初期フレームは 0 報酬で 1 枚記録される。
        # step 毎の rewards が [+0.5, -0.5] -> [+1.0, +0.0] -> [-2.0, +3.0] のとき、
        # 表示用に渡される累計は [0.0, 0.0] -> [+0.5, -0.5] -> [+1.5, -0.5] -> [-0.5, +2.5] になるはず。
        step_infos = [
            {"rewards": [0.5, -0.5]},
            {"rewards": [1.0, 0.0]},
            {"rewards": [-2.0, 3.0]},
        ]
        env = _FakeRenderableEnv(step_infos)
        trainer = _FakeTrainer()
        value_gif = _RecordingValueGif()

        train_ppo.run_episode(env, trainer, buffer=None, value_gif=value_gif)

        self.assertEqual(len(value_gif.recorded_rewards), 4)
        np.testing.assert_allclose(value_gif.recorded_rewards[0], [0.0, 0.0])
        np.testing.assert_allclose(value_gif.recorded_rewards[1], [0.5, -0.5])
        np.testing.assert_allclose(value_gif.recorded_rewards[2], [1.5, -0.5])
        np.testing.assert_allclose(value_gif.recorded_rewards[3], [-0.5, 2.5])

    def test_value_gif_cumulative_rewards_are_independent_of_recorded_values(self):
        # values は trainer から来るので毎 step 上書きされるだけ。reward 側は累積されることを担保する。
        step_infos = [
            {"rewards": [0.1, 0.2]},
            {"rewards": [0.1, 0.2]},
        ]
        env = _FakeRenderableEnv(step_infos)
        trainer = _FakeTrainer()
        value_gif = _RecordingValueGif()

        train_ppo.run_episode(env, trainer, buffer=None, value_gif=value_gif)

        np.testing.assert_allclose(value_gif.recorded_rewards[0], [0.0, 0.0])
        np.testing.assert_allclose(value_gif.recorded_rewards[1], [0.1, 0.2])
        np.testing.assert_allclose(value_gif.recorded_rewards[2], [0.2, 0.4])


@unittest.skipUnless(HAS_TORCH, "torch is required")
class RunEpisodeAggregationTest(unittest.TestCase):
    def test_run_episode_reports_cumulative_auxiliary_rewards(self):
        # 3 step ぶんの補助報酬が、最後の step に依存せず合計として記録されることを確認する。
        step_infos = [
            {
                "missile_tracking_penalty": 1.0,
                "nearest_enemy_facing_reward": 0.2,
                "nearest_enemy_facing_penalty": 0.1,
                "low_movement_penalty": 0.0,
                "missile_fire_reward": 0.5,
                "kill_reward": 0.0,
                "death_penalty": 0.0,
                "mean_movement_distance_1s": 5.0,
                "out_of_bounds_penalty": 0.0,
                "blue_kills": 0.0,
                "blue_losses": 0.0,
                "hit_events": 0.0,
                "blue_alive": 1.0,
                "red_alive": 1.0,
                "outcome": 0.0,
                "reward_mean": 1.6,
            },
            {
                "missile_tracking_penalty": 0.0,
                "nearest_enemy_facing_reward": 0.0,
                "nearest_enemy_facing_penalty": 0.3,
                "low_movement_penalty": 0.2,
                "missile_fire_reward": 0.0,
                "kill_reward": 10.0,
                "death_penalty": 0.0,
                "mean_movement_distance_1s": 6.0,
                "out_of_bounds_penalty": 0.0,
                "blue_kills": 1.0,
                "blue_losses": 0.0,
                "hit_events": 1.0,
                "blue_alive": 1.0,
                "red_alive": 0.0,
                "outcome": 1.0,
                "reward_mean": 10.2,
            },
            {
                "missile_tracking_penalty": 0.0,
                "nearest_enemy_facing_reward": 0.1,
                "nearest_enemy_facing_penalty": 0.0,
                "low_movement_penalty": 0.0,
                "missile_fire_reward": 0.0,
                "kill_reward": 0.0,
                "death_penalty": 20.0,
                "mean_movement_distance_1s": 0.0,
                "out_of_bounds_penalty": 0.0,
                "blue_kills": 0.0,
                "blue_losses": 1.0,
                "hit_events": 1.0,
                "terminal_score": -2.0,
                "blue_alive": 0.0,
                "red_alive": 1.0,
                "outcome": -1.0,
                "reward_mean": -22.0,
            },
        ]
        env = _FakeEnv(step_infos)
        trainer = _FakeTrainer()

        _, total_reward, info = train_ppo.run_episode(env, trainer, buffer=None)

        # エピソード合計の補助報酬が現れる。
        self.assertAlmostEqual(info["missile_tracking_penalty"], 1.0)
        self.assertAlmostEqual(info["nearest_enemy_facing_reward"], 0.2 + 0.1)
        self.assertAlmostEqual(info["nearest_enemy_facing_penalty"], 0.1 + 0.3)
        self.assertAlmostEqual(info["low_movement_penalty"], 0.2)
        self.assertAlmostEqual(info["missile_fire_reward"], 0.5)
        self.assertAlmostEqual(info["kill_reward"], 10.0)
        self.assertAlmostEqual(info["death_penalty"], 20.0)
        self.assertAlmostEqual(info["blue_kills"], 1.0)
        self.assertAlmostEqual(info["blue_losses"], 1.0)
        self.assertAlmostEqual(info["hit_events"], 2.0)
        self.assertAlmostEqual(info["mean_movement_distance_1s"], (5.0 + 6.0 + 0.0) / 3.0)
        # 終端値はそのまま最後の step の値が残る。
        self.assertAlmostEqual(info["terminal_score"], -2.0)
        self.assertAlmostEqual(info["outcome"], -1.0)
        self.assertAlmostEqual(info["episode_steps"], 3.0)
        self.assertAlmostEqual(total_reward, 1.6 + 10.2 + -22.0, places=5)


class _FakeEnv:
    def __init__(self, step_infos):
        self._step_infos = step_infos
        self.max_steps = max(len(step_infos), 1)
        self._step_index = 0

    def reset(self):
        self._step_index = 0
        return np.zeros((1, 1), dtype=np.float32)

    def step(self, _actions):
        info = self._step_infos[self._step_index]
        done = self._step_index == len(self._step_infos) - 1
        self._step_index += 1
        rewards = np.array([float(info["reward_mean"])], dtype=np.float32)
        return StepResult(
            observations=np.zeros((1, 1), dtype=np.float32),
            rewards=rewards,
            done=done,
            info=dict(info),
        )

    def take_render_frame(self):
        return None


class _FakeTrainer:
    def reset_exploration_noise(self):
        return None

    def act(self, observations, deterministic=False):
        agent_count = int(observations.shape[0])
        raw_actions = np.zeros((agent_count, 3), dtype=np.float32)
        env_actions = np.zeros((agent_count, 3), dtype=np.float32)
        log_probs = np.zeros((agent_count,), dtype=np.float32)
        values = np.zeros((agent_count,), dtype=np.float32)
        return raw_actions, env_actions, log_probs, values


class _FakeRenderableEnv:
    """take_render_frame が常にダミーのフレームを返すテスト用環境。"""

    def __init__(self, step_infos):
        self._step_infos = step_infos
        self.max_steps = max(len(step_infos), 1)
        self._step_index = 0
        self._agent_count = len(step_infos[0]["rewards"]) if step_infos else 1

    def reset(self):
        self._step_index = 0
        return np.zeros((self._agent_count, 1), dtype=np.float32)

    def step(self, _actions):
        info = self._step_infos[self._step_index]
        done = self._step_index == len(self._step_infos) - 1
        self._step_index += 1
        rewards = np.array(info["rewards"], dtype=np.float32)
        return StepResult(
            observations=np.zeros((self._agent_count, 1), dtype=np.float32),
            rewards=rewards,
            done=done,
            info={"reward_mean": float(np.mean(rewards))},
        )

    def take_render_frame(self):
        return np.zeros((4, 4, 4), dtype=np.uint8)


class _RecordingValueGif:
    def __init__(self):
        self.recorded_values = []
        self.recorded_rewards = []

    def record(self, frame, values, rewards):
        self.recorded_values.append(np.asarray(values, dtype=np.float32).copy())
        self.recorded_rewards.append(np.asarray(rewards, dtype=np.float32).copy())


if __name__ == "__main__":
    unittest.main()
