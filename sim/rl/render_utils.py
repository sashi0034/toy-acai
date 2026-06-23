from pathlib import Path
from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

from .observation import ObservationFeature


HIGHLIGHT_TEXT_COLOR = (255, 220, 120, 235)


def render_reward(
    frame: Image.Image, total_reward: float, critic_value: float | None = None
) -> Image.Image:
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    text = f"reward={total_reward:+.3f}"
    if critic_value is not None:
        text += f"  critic={critic_value:+.3f}"
    text_box = draw.textbbox((0, 0), text, font=font)
    padding = 6
    margin = 8
    line_height = text_box[3]
    panel_width = text_box[2] - text_box[0] + padding * 2

    # Reserve the second line for the final total reward, which is known only
    # after the episode finishes.
    panel_height = line_height * 2 + padding * 2

    draw.rounded_rectangle(
        (margin, margin, margin + panel_width, margin + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )
    draw.text(
        (margin + padding, margin + padding),
        text,
        fill=(255, 255, 255, 235),
        font=font,
    )
    return Image.alpha_composite(image, overlay)


def render_actual_total_reward(frame: Image.Image, total_reward: float) -> Image.Image:
    """Draw the delayed-reward-inclusive total attributed to this frame."""
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    text = f"reward={total_reward:+.3f}"
    text_box = draw.textbbox((0, 0), text, font=font)
    padding = 6
    margin = 8
    line_height = text_box[3]

    draw.text(
        (margin + padding, margin + padding + line_height),
        text,
        fill=HIGHLIGHT_TEXT_COLOR,
        font=font,
    )
    return Image.alpha_composite(image, overlay)


def render_observation(
    frame: Image.Image, features: Sequence[ObservationFeature]
) -> Image.Image:
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    groups: list[tuple[str, list[ObservationFeature]]] = []
    for feature in features:
        if not groups or groups[-1][0] != feature.group:
            groups.append((feature.group, []))
        groups[-1][1].append(feature)

    padding = 6
    margin = 8
    group_gap = 4
    column_gap = 8
    line_height = draw.textbbox((0, 0), "+0.000", font=font)[3]
    text_width = lambda text: draw.textbbox((0, 0), text, font=font)[2]
    label_width = max(text_width(feature.name) for feature in features)
    value_width = max(text_width(f"{feature.value:+.3f}") for feature in features)
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

    draw.rounded_rectangle(
        (panel_x, margin, panel_x + panel_width, margin + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )

    y = margin + padding
    for group_index, (name, group_features) in enumerate(groups):
        if group_index:
            draw.line(
                (
                    panel_x + padding,
                    y - group_gap // 2,
                    panel_x + panel_width - padding,
                    y - group_gap // 2,
                ),
                fill=(255, 255, 255, 100),
            )
        draw.text((panel_x + padding, y), name, fill=HIGHLIGHT_TEXT_COLOR, font=font)
        y += line_height
        for feature in group_features:
            draw.text(
                (panel_x + padding, y),
                feature.name,
                fill=(255, 255, 255, 235),
                font=font,
            )
            draw.text(
                (panel_x + panel_width - padding, y),
                f"{feature.value:+.3f}",
                anchor="ra",
                fill=(255, 255, 255, 235),
                font=font,
            )
            y += line_height
        y += group_gap

    return Image.alpha_composite(image, overlay)


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
