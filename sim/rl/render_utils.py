from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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
