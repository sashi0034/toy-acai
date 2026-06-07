#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np


def add_default_module_paths(repo_root):
    for path in (
        repo_root / "linux-python" / "build",
        repo_root / "build",
    ):
        if path.exists():
            sys.path.insert(0, str(path))


def make_actions(rng, step, fighter_count):
    actions = np.zeros((fighter_count, 3), dtype=np.float64)
    actions[:, 0] = rng.uniform(0.2, 1.0, size=fighter_count)
    actions[:, 1] = rng.uniform(-0.7, 0.7, size=fighter_count)
    actions[:, 2] = 1.0 if step % 12 == 0 else 0.0
    return actions


def fighter_triangle(x, y, yaw, size):
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-forward[1], forward[0]])
    center = np.array([x, y])
    nose = center + forward * size
    left = center - forward * size * 0.65 - right * size * 0.55
    right_point = center - forward * size * 0.65 + right * size * 0.55
    return [tuple(nose), tuple(left), tuple(right_point)]


def render_gif(frames, output_path, dt, size=(960, 540)):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow is required only for --gif. Install it with: python3.11 -m pip install --user Pillow") from exc

    images = []
    width, height = size
    colors = [(36, 118, 219), (220, 54, 67)]

    for obs in frames:
        image = Image.new("RGB", size, (248, 248, 246))
        draw = ImageDraw.Draw(image)

        battlefield = obs["battlefield"]
        fighters = obs["fighters"]
        missiles = obs["missiles"]
        _, _, field_w, field_h = battlefield
        scale = min((width - 40) / field_w, (height - 40) / field_h)
        origin = ((width - field_w * scale) * 0.5, (height - field_h * scale) * 0.5)

        def to_screen(px, py):
            return origin[0] + px * scale, origin[1] + py * scale

        rect = [origin[0], origin[1], origin[0] + field_w * scale, origin[1] + field_h * scale]
        draw.rectangle(rect, outline=(35, 35, 35), width=2)

        grid_step = 100
        for gx in np.arange(0, field_w + 1, grid_step):
            x0, y0 = to_screen(gx, 0)
            x1, y1 = to_screen(gx, field_h)
            draw.line((x0, y0, x1, y1), fill=(224, 224, 224))
        for gy in np.arange(0, field_h + 1, grid_step):
            x0, y0 = to_screen(0, gy)
            x1, y1 = to_screen(field_w, gy)
            draw.line((x0, y0, x1, y1), fill=(224, 224, 224))

        for missile in missiles:
            x, y = to_screen(missile[0], missile[1])
            team_id = int(missile[6])
            color = colors[team_id % len(colors)]
            r = 4
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

        for fighter in fighters:
            if fighter[6] <= 0.0:
                continue
            team_id = int(fighter[0])
            x, y = to_screen(fighter[2], fighter[3])
            yaw = fighter[4]
            points = fighter_triangle(x, y, yaw, 13)
            draw.polygon(points, fill=colors[team_id % len(colors)], outline=(20, 20, 20))

        images.append(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(dt * 1000)))
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Run a random headless toy-acai simulation from Python.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/states.npz"))
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--module-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.module_dir is not None:
        sys.path.insert(0, str(args.module_dir))
    add_default_module_paths(repo_root)

    import toy_acai_core

    rng = np.random.default_rng(args.seed)
    env = toy_acai_core.BattlefieldEnv()
    obs = env.reset()
    frames = [obs]
    fighter_history = [np.array(obs["fighters"], copy=True)]
    missile_counts = [len(obs["missiles"])]

    for step in range(args.steps):
        actions = make_actions(rng, step, toy_acai_core.FIGHTER_COUNT)
        obs = env.step(actions, args.dt)
        frames.append(obs)
        fighter_history.append(np.array(obs["fighters"], copy=True))
        missile_counts.append(len(obs["missiles"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fighters=np.stack(fighter_history),
        missile_counts=np.array(missile_counts, dtype=np.int32),
        battlefield=np.array(frames[0]["battlefield"], dtype=np.float64),
        screen_size=np.array(frames[0]["screen_size"], dtype=np.float64),
    )
    print("saved", args.output)

    if args.gif is not None:
        render_gif(frames, args.gif, args.dt)
        print("saved", args.gif)


if __name__ == "__main__":
    main()
