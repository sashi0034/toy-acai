from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from .observation import ObservationFeature


HIGHLIGHT_TEXT_COLOR = (255, 220, 120, 235)
PANEL_PADDING = 6
PANEL_MARGIN = 8
REWARD_VALUE_FORMAT = "+.3f"
OBSERVATION_VALUE_FORMAT = "+8.3f"


@dataclass
class _RewardPanelLayout:
    panel: Image.Image
    panel_position: tuple[int, int]
    first_label_position: tuple[int, int]
    first_value_position: tuple[int, int]
    second_label_position: tuple[int, int]
    second_value_position: tuple[int, int]
    first_line_y: int
    second_line_y: int
    text_layer_size: tuple[int, int]


@dataclass
class _ObservationPanelLayout:
    image_size: tuple[int, int]
    signature: tuple[tuple[str, str], ...]
    panel: Image.Image
    panel_position: tuple[int, int]
    value_positions: tuple[tuple[int, int], ...]


_observation_panel_layout: _ObservationPanelLayout | None = None
_reward_panel_layout: _RewardPanelLayout | None = None


def _ensure_rgba(frame: Image.Image) -> Image.Image:
    if frame.mode == "RGBA":
        return frame
    print(f"[WARN] converting render frame from {frame.mode} to RGBA")
    return frame.convert("RGBA")


def _get_reward_panel_layout(font: ImageFont.ImageFont) -> _RewardPanelLayout:
    global _reward_panel_layout

    if _reward_panel_layout is not None:
        return _reward_panel_layout

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    label_width = max(
        measure.textbbox((0, 0), "reward=", font=font)[2],
        measure.textbbox((0, 0), "critic=", font=font)[2],
        measure.textbbox((0, 0), "return=", font=font)[2],
    )
    value_width = max(
        measure.textbbox((0, 0), f"{-999.999:{REWARD_VALUE_FORMAT}}", font=font)[2],
        measure.textbbox((0, 0), f"{999.999:{REWARD_VALUE_FORMAT}}", font=font)[2],
    )
    line_height = measure.textbbox(
        (0, 0), f"{999.999:{REWARD_VALUE_FORMAT}}", font=font
    )[3]
    column_gap = 12
    panel_width = label_width * 2 + value_width * 2 + column_gap + PANEL_PADDING * 2
    panel_height = line_height * 2 + PANEL_PADDING * 2
    first_label_x = PANEL_PADDING
    first_value_x = first_label_x + label_width
    second_label_x = first_value_x + value_width + column_gap
    second_value_x = second_label_x + label_width
    first_line_y = PANEL_PADDING
    second_line_y = PANEL_PADDING + line_height

    panel = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rounded_rectangle(
        (0, 0, panel_width, panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )

    _reward_panel_layout = _RewardPanelLayout(
        panel=panel,
        panel_position=(PANEL_MARGIN, PANEL_MARGIN),
        first_label_position=(first_label_x, first_line_y),
        first_value_position=(first_value_x, first_line_y),
        second_label_position=(second_label_x, first_line_y),
        second_value_position=(second_value_x, first_line_y),
        first_line_y=first_line_y,
        second_line_y=second_line_y,
        text_layer_size=(panel_width, panel_height),
    )
    return _reward_panel_layout


def render_reward(
    frame: Image.Image, total_reward: float, critic_value: float | None = None
) -> Image.Image:
    image = _ensure_rgba(frame)
    font = ImageFont.load_default()
    layout = _get_reward_panel_layout(font)
    image.alpha_composite(layout.panel, layout.panel_position)

    text_layer = Image.new("RGBA", layout.text_layer_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")
    text_color = (255, 255, 255, 235)
    first_label_x, _ = layout.first_label_position
    first_value_x, _ = layout.first_value_position
    second_label_x, _ = layout.second_label_position
    second_value_x, _ = layout.second_value_position
    draw.text(
        (first_label_x, layout.first_line_y),
        "reward=",
        fill=text_color,
        font=font,
    )
    draw.text(
        (first_value_x, layout.first_line_y),
        f"{total_reward:{REWARD_VALUE_FORMAT}}",
        fill=text_color,
        font=font,
    )
    if critic_value is not None:
        draw.text(
            (second_label_x, layout.first_line_y),
            "critic=",
            fill=text_color,
            font=font,
        )
        draw.text(
            (second_value_x, layout.first_line_y),
            f"{critic_value:{REWARD_VALUE_FORMAT}}",
            fill=text_color,
            font=font,
        )
    image.alpha_composite(text_layer, layout.panel_position)
    return image


def render_actual_total_reward(
    frame: Image.Image, total_reward: float, returns: float
) -> Image.Image:
    """Draw the delayed-reward-inclusive reward and return for this frame."""
    image = _ensure_rgba(frame)
    font = ImageFont.load_default()
    layout = _get_reward_panel_layout(font)

    text_layer = Image.new("RGBA", layout.text_layer_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer, "RGBA")
    first_label_x, _ = layout.first_label_position
    first_value_x, _ = layout.first_value_position
    second_label_x, _ = layout.second_label_position
    second_value_x, _ = layout.second_value_position
    draw.text(
        (first_label_x, layout.second_line_y),
        "reward=",
        fill=HIGHLIGHT_TEXT_COLOR,
        font=font,
    )
    draw.text(
        (first_value_x, layout.second_line_y),
        f"{total_reward:{REWARD_VALUE_FORMAT}}",
        fill=HIGHLIGHT_TEXT_COLOR,
        font=font,
    )
    draw.text(
        (second_label_x, layout.second_line_y),
        "return=",
        fill=HIGHLIGHT_TEXT_COLOR,
        font=font,
    )
    draw.text(
        (second_value_x, layout.second_line_y),
        f"{returns:{REWARD_VALUE_FORMAT}}",
        fill=HIGHLIGHT_TEXT_COLOR,
        font=font,
    )
    image.alpha_composite(text_layer, layout.panel_position)
    return image


def _observation_panel_signature(
    features: Sequence[ObservationFeature],
) -> tuple[tuple[str, str], ...]:
    return tuple((feature.group, feature.name) for feature in features)


def _get_observation_panel_layout(
    image: Image.Image,
    features: Sequence[ObservationFeature],
    font: ImageFont.ImageFont,
) -> _ObservationPanelLayout | None:
    global _observation_panel_layout

    if not features:
        return None

    signature = _observation_panel_signature(features)
    image_size = image.size
    if (
        _observation_panel_layout is not None
        and _observation_panel_layout.image_size == image_size
        and _observation_panel_layout.signature == signature
    ):
        return _observation_panel_layout

    groups: list[tuple[str, list[ObservationFeature]]] = []
    for feature in features:
        if not groups or groups[-1][0] != feature.group:
            groups.append((feature.group, []))
        groups[-1][1].append(feature)

    padding = 6
    margin = 8
    group_gap = 4
    column_gap = 8

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    line_height = measure.textbbox((0, 0), "+0.000", font=font)[3]

    def text_width(text: str) -> int:
        return int(measure.textbbox((0, 0), text, font=font)[2])

    label_width = max(text_width(feature.name) for feature in features)
    value_width = max(
        text_width(f"{-999.999:{OBSERVATION_VALUE_FORMAT}}"),
        text_width(f"{999.999:{OBSERVATION_VALUE_FORMAT}}"),
    )
    panel_width = max(
        max(text_width(group) for group, _ in groups),
        label_width + column_gap + value_width,
    )
    panel_width += padding * 2
    panel_height = (
        (len(groups) + len(features)) * line_height
        + (len(groups) - 1) * group_gap
        + padding * 2
    )
    panel_x = image.width - margin - panel_width
    panel_y = margin

    panel = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rounded_rectangle(
        (0, 0, panel_width, panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )

    value_positions: list[tuple[int, int]] = []
    y = padding
    for group_index, (name, group_features) in enumerate(groups):
        if group_index:
            draw.line(
                (
                    padding,
                    y - group_gap // 2,
                    panel_width - padding,
                    y - group_gap // 2,
                ),
                fill=(255, 255, 255, 100),
            )
        draw.text((padding, y), name, fill=HIGHLIGHT_TEXT_COLOR, font=font)
        y += line_height
        for feature in group_features:
            draw.text(
                (padding, y),
                feature.name,
                fill=(255, 255, 255, 235),
                font=font,
            )
            value_positions.append((panel_x + panel_width - padding, panel_y + y))
            y += line_height
        y += group_gap

    _observation_panel_layout = _ObservationPanelLayout(
        image_size=image_size,
        signature=signature,
        panel=panel,
        panel_position=(panel_x, panel_y),
        value_positions=tuple(value_positions),
    )
    return _observation_panel_layout


def render_observation(
    frame: Image.Image, features: Sequence[ObservationFeature]
) -> Image.Image:
    image = _ensure_rgba(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    layout = _get_observation_panel_layout(image, features, font)
    if layout is None:
        return image

    image.alpha_composite(layout.panel, layout.panel_position)
    for feature, value_position in zip(features, layout.value_positions, strict=True):
        draw.text(
            value_position,
            f"{feature.value:{OBSERVATION_VALUE_FORMAT}}",
            anchor="ra",
            fill=(255, 255, 255, 235),
            font=font,
        )

    return image


def save_rendered_frames(
    frames: list[Image.Image], render_path: Path, interval_seconds: float
):
    if not frames:
        print(f"No frames to save: {render_path}")
        return

    frames[0].save(
        render_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=round(interval_seconds * 1000),
        loop=0,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames to {render_path}")
