import importlib.util
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
                    {"episode": 10, "reward": -2.5},
                    {"episode": 20, "reward": -1.0},
                    {"episode": 30, "reward": 0.75},
                ],
            )

            self.assertTrue(path.exists())
            with Image.open(path) as image:
                self.assertEqual(image.size, (900, 520))
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
