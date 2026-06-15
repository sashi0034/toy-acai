import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

HAS_PIL = find_spec("PIL") is not None
if HAS_PIL:
    from PIL import Image

    from toy_acai_rl.value_gif import (  # noqa: E402
        ValueGifRecorder,
        draw_value_overlay,
        value_overlay_lines,
    )


@unittest.skipUnless(HAS_PIL, "Pillow is not installed")
class ValueGifTest(unittest.TestCase):
    def test_value_overlay_lines_use_blue_fighter_labels(self):
        self.assertEqual(
            value_overlay_lines([1.0, -2.5, 0.0, 3.25]),
            [
                "B0 value=+1.000",
                "B1 value=-2.500",
                "B2 value=+0.000",
                "B3 value=+3.250",
            ],
        )

    def test_draw_value_overlay_preserves_image_size(self):
        frame = np.full((32, 48, 4), 255, dtype=np.uint8)

        image = draw_value_overlay(frame, [0.1, 0.2, 0.3, 0.4])

        self.assertEqual(image.size, (48, 32))

    def test_recorder_saves_gif_with_recorded_frames(self):
        frame_a = np.full((32, 48, 4), 255, dtype=np.uint8)
        frame_b = np.zeros((32, 48, 4), dtype=np.uint8)
        frame_b[:, :, 3] = 255

        with tempfile.TemporaryDirectory() as tmp_dir:
            gif_path = Path(tmp_dir) / "value.gif"
            recorder = ValueGifRecorder(gif_path, render_interval=0.1)
            recorder.record(frame_a, [0.1, 0.2, 0.3, 0.4])
            recorder.record(frame_b, [-0.1, -0.2, -0.3, -0.4])
            recorder.save()

            with Image.open(gif_path) as image:
                self.assertEqual(image.size, (48, 32))
                self.assertEqual(getattr(image, "n_frames", 1), 2)


if __name__ == "__main__":
    unittest.main()
