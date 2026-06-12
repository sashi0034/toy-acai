#!/usr/bin/env python3
import argparse
import os
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


def main():
    parser = argparse.ArgumentParser(description="Run a random headless toy-acai simulation from Python.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/states.npz"))
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--module-dir", type=Path, default=None)
    args = parser.parse_args()

    original_cwd = Path.cwd()
    if not args.output.is_absolute():
        args.output = original_cwd / args.output
    if args.gif is not None and not args.gif.is_absolute():
        args.gif = original_cwd / args.gif

    repo_root = Path(__file__).resolve().parents[1]
    module_dir = args.module_dir.resolve() if args.module_dir is not None else repo_root / "linux-python" / "build"
    if args.module_dir is not None:
        sys.path.insert(0, str(module_dir))
    add_default_module_paths(repo_root)

    import toy_acai_core

    rng = np.random.default_rng(args.seed)
    if args.gif is not None:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        if (module_dir / "resources").exists():
            os.chdir(module_dir)
    env = toy_acai_core.BattlefieldEnv(
        render=args.gif is not None,
        gif_path=str(args.gif) if args.gif is not None else "",
    )
    obs = env.reset()
    initial_obs = obs
    fighter_history = [np.array(obs["fighters"], copy=True)]
    missile_counts = [len(obs["missiles"])]

    for step in range(args.steps):
        actions = make_actions(rng, step, toy_acai_core.FIGHTER_COUNT)
        obs = env.step(actions, args.dt)
        fighter_history.append(np.array(obs["fighters"], copy=True))
        missile_counts.append(len(obs["missiles"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        fighters=np.stack(fighter_history),
        missile_counts=np.array(missile_counts, dtype=np.int32),
        battlefield=np.array(initial_obs["battlefield"], dtype=np.float64),
        screen_size=np.array(initial_obs["screen_size"], dtype=np.float64),
    )
    print("saved", args.output)

    if args.gif is not None:
        env.close_gif()
        print("saved", args.gif)


if __name__ == "__main__":
    main()
