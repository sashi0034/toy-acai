import os
import sys
import unittest
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "linux-python" / "build"
HAS_CURRENT_PYTHON_MODULE = any(
    (MODULE_DIR / f"toy_acai_core{suffix}").exists() for suffix in EXTENSION_SUFFIXES
)


@unittest.skipUnless(HAS_CURRENT_PYTHON_MODULE, "toy_acai_core for this Python is missing")
class RenderFrameTest(unittest.TestCase):
    def test_take_render_frame_returns_uint8_rgba_frame_after_render_interval(self):
        sys.path.insert(0, str(MODULE_DIR))
        import toy_acai_core  # noqa: E402

        if not hasattr(toy_acai_core.BattlefieldEnv, "take_render_frame"):
            self.skipTest("toy_acai_core was not rebuilt with take_render_frame")

        original_cwd = Path.cwd()
        if (MODULE_DIR / "resources").exists():
            os.chdir(MODULE_DIR)
        try:
            env = toy_acai_core.BattlefieldEnv(
                render=True,
                render_width=96,
                render_height=54,
                render_interval=toy_acai_core.SIMULATION_DELTA_TIME,
            )
            env.reset()
            actions = np.zeros((toy_acai_core.FIGHTER_COUNT, 3), dtype=np.float64)
            env.step(actions)
            frame = env.take_render_frame()
        finally:
            os.chdir(original_cwd)

        self.assertIsNotNone(frame)
        self.assertEqual(frame.dtype, np.uint8)
        self.assertEqual(frame.shape, (54, 96, 4))


if __name__ == "__main__":
    unittest.main()
