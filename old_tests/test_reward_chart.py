import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_PIL:
    from PIL import Image

if HAS_TORCH:
    import train_ppo  # noqa: E402


@unittest.skipUnless(HAS_PIL and HAS_TORCH, "Pillow and torch are required")
class RewardChartTest(unittest.TestCase):
    def test_make_reward_chart_image_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "reward_chart.png"
            train_ppo.make_reward_chart_image(
                path,
                [
                    {"episode": 10, "reward": -2.5, "opponent_count": 2},
                    {"episode": 20, "reward": -1.0, "opponent_count": 2},
                    {"episode": 30, "reward": 0.75, "opponent_count": 2},
                ],
                stage_label="red=2",
            )

            self.assertTrue(path.exists())
            with Image.open(path) as image:
                self.assertEqual(image.size, (900, 520))
                self.assertEqual(image.format, "PNG")

    def test_reward_history_filters_to_current_opponent_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "eval_metrics.jsonl"
            rows = [
                {"episode": 10, "reward": -1.0, "opponent_count": 1, "curriculum_stage": 1},
                {"episode": 20, "reward": -2.0, "opponent_count": 2, "curriculum_stage": 2},
                {"episode": 30, "reward": -3.0, "opponent_count": 2, "curriculum_stage": 2},
                {"episode": 40, "reward": -4.0, "opponent_count": 3, "curriculum_stage": 3},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            history = train_ppo.reward_history_from_jsonl(path, max_episode=35, opponent_count=2)

            self.assertEqual(
                history,
                [
                    {"episode": 20, "reward": -2.0, "opponent_count": 2, "curriculum_stage": 2},
                    {"episode": 30, "reward": -3.0, "opponent_count": 2, "curriculum_stage": 2},
                ],
            )

    def test_reward_chart_stage_reset_clears_history_and_counter(self):
        history = [{"episode": 10, "reward": -1.0, "opponent_count": 1}]

        next_count = train_ppo.reset_reward_chart_stage(history)

        self.assertEqual(history, [])
        self.assertEqual(next_count, 0)

    def test_reward_chart_spool_comment_includes_red_count(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            chart = root / "chart.png"
            chart.write_bytes(b"fake")

            train_ppo.make_reward_chart_spool_record(
                root,
                chart,
                [{"episode": 20, "reward": -2.0, "opponent_count": 2}],
            )

            records = list((root / "pending").glob("reward_chart_*.json"))
            self.assertEqual(len(records), 1)
            payload = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertIn("red=2", payload["comment"])


if __name__ == "__main__":
    unittest.main()
