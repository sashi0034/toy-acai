from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .observation import ENTITY_FEATURES, MISSILE_FEATURES, SELF_FEATURES


def render_reward(frame: Image.Image, total_reward: float) -> Image.Image:
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    text = f"reward={total_reward:+.3f}"
    text_box = draw.textbbox((0, 0), text, font=font)
    padding = 6
    margin = 8
    panel_width = text_box[2] - text_box[0] + padding * 2
    panel_height = text_box[3] - text_box[1] + padding * 2

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


def render_observation(frame: Image.Image, observation) -> Image.Image:
    image = frame.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    observation = observation.detach().cpu().flatten().tolist()
    groups = (
        ("SELF_FEATURES", SELF_FEATURES),
        ("ENTITY_FEATURES[0]", ENTITY_FEATURES),
        ("ENTITY_FEATURES[1]", ENTITY_FEATURES),
        ("MISSILE_FEATURES[0]", MISSILE_FEATURES),
        ("MISSILE_FEATURES[1]", MISSILE_FEATURES),
    )
    assert sum(size for _, size in groups) == len(observation)

    padding = 6
    margin = 8
    group_gap = 4
    line_height = draw.textbbox((0, 0), "+0.000", font=font)[3]
    text_width = max(
        draw.textbbox((0, 0), text, font=font)[2]
        for text in [name for name, _ in groups]
        + [f"{value:+.3f}" for value in observation]
    )
    panel_width = text_width + padding * 2
    panel_height = (
        (len(groups) + len(observation)) * line_height
        + (len(groups) - 1) * group_gap
        + padding * 2
    )
    panel_x = image.width - margin - panel_width

    draw.rounded_rectangle(
        (panel_x, margin, panel_x + panel_width, margin + panel_height),
        radius=4,
        fill=(0, 0, 0, 150),
    )

    value_index = 0
    y = margin + padding
    for group_index, (name, size) in enumerate(groups):
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
        draw.text(
            (panel_x + padding, y), name, fill=(255, 220, 120, 235), font=font
        )
        y += line_height
        for value in observation[value_index : value_index + size]:
            draw.text(
                (panel_x + padding, y),
                f"{value:+.3f}",
                fill=(255, 255, 255, 235),
                font=font,
            )
            y += line_height
        value_index += size
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
