from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def value_overlay_lines(values: Iterable[float]) -> List[str]:
    return [f"B{agent_id} value={float(value):+.3f}" for agent_id, value in enumerate(values)]


def draw_value_overlay(frame: np.ndarray, values: Iterable[float]) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    lines = value_overlay_lines(values)
    if not lines:
        return image

    padding = 6
    margin = 8
    line_gap = 2
    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in text_boxes)
    line_height = max(box[3] - box[1] for box in text_boxes)
    panel_width = min(image.width - margin * 2, text_width + padding * 2)
    panel_height = min(
        image.height - margin * 2,
        line_height * len(lines) + line_gap * (len(lines) - 1) + padding * 2,
    )
    if panel_width <= 0 or panel_height <= 0:
        return image

    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )
    y = margin + padding
    for line in lines:
        if y + line_height > margin + panel_height:
            break
        draw.text((margin + padding, y), line, fill=(255, 255, 255, 235), font=font)
        y += line_height + line_gap

    return Image.alpha_composite(image, overlay)


class FrameGifRecorder:
    def __init__(self, path: Path, render_interval: float):
        self.path = Path(path)
        self.duration_ms = max(1, int(round(render_interval * 1000.0)))
        self.frames: List[Image.Image] = []

    def record(self, frame: np.ndarray) -> None:
        self.frames.append(Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB"))

    def save(self) -> None:
        if not self.frames:
            raise RuntimeError(f"no frames were recorded for GIF: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        first, *rest = self.frames
        first.save(
            self.path,
            save_all=True,
            append_images=rest,
            duration=self.duration_ms,
            loop=0,
        )


class ValueGifRecorder(FrameGifRecorder):
    def record(self, frame: np.ndarray, values: Iterable[float]) -> None:
        self.frames.append(draw_value_overlay(frame, values).convert("RGB"))
