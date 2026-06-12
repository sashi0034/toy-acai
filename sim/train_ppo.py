#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from toy_acai_rl.env import ToyAcaiPPOEnv, load_core, observation_dim
from toy_acai_rl.ppo import PPOConfig, PPOTrainer, RolloutBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO policy for the toy-acai simulator.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rl/default"))
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--module-dir", type=Path, default=None)
    parser.add_argument("--slack-spool", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def make_spool_record(spool_root: Path, gif_path: Path, metrics: dict) -> None:
    pending = spool_root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    record_id = f"episode_{int(metrics['episode']):06d}_{int(time.time())}"
    tmp_path = pending / f".{record_id}.tmp"
    final_path = pending / f"{record_id}.json"
    payload = {
        "gif_path": str(gif_path.resolve()),
        "episode": int(metrics["episode"]),
        "reward": float(metrics["reward"]),
        "outcome": float(metrics["outcome"]),
        "blue_alive": float(metrics["blue_alive"]),
        "red_alive": float(metrics["red_alive"]),
        "comment": f"toy-acai PPO episode {int(metrics['episode'])}: reward={metrics['reward']:.3f}, outcome={metrics['outcome']:+.0f}",
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(final_path)


def run_episode(env: ToyAcaiPPOEnv, trainer: PPOTrainer, buffer: Optional[RolloutBuffer], deterministic: bool = False):
    observations = env.reset()
    total_reward = 0.0
    final_info = {}
    for _ in range(env.max_steps):
        raw_actions, env_actions, log_probs, values = trainer.act(observations, deterministic=deterministic)
        result = env.step(env_actions)
        if buffer is not None:
            buffer.add(observations, raw_actions, log_probs, result.rewards, result.done, values)
        total_reward += float(np.mean(result.rewards))
        observations = result.observations
        final_info = result.info
        if result.done:
            break
    return observations, total_reward, final_info


def evaluate(toy_acai_core, trainer: PPOTrainer, args, episode: int, repo_root: Path):
    media_dir = args.out_dir / "media"
    gif_path = media_dir / f"episode_{episode:06d}.gif"
    module_dir = args.module_dir.resolve() if args.module_dir is not None else repo_root / "linux-python" / "build"
    original_cwd = Path.cwd()
    env = ToyAcaiPPOEnv(
        toy_acai_core,
        max_steps=args.steps,
        render=True,
        gif_path=gif_path.resolve(),
        module_dir=module_dir,
    )
    try:
        _, reward, info = run_episode(env, trainer, buffer=None, deterministic=True)
    finally:
        env.close()
        os.chdir(original_cwd)

    metrics = {
        "episode": episode,
        "reward": reward,
        "blue_alive": info.get("blue_alive", 0.0),
        "red_alive": info.get("red_alive", 0.0),
        "outcome": info.get("outcome", 0.0),
        "gif": str(gif_path),
    }
    write_jsonl(args.out_dir / "eval_metrics.jsonl", metrics)
    if args.slack_spool is not None:
        make_spool_record(args.slack_spool, gif_path, metrics)
    return metrics


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir = args.out_dir.resolve()
    if args.module_dir is not None:
        args.module_dir = args.module_dir.resolve()
    if args.slack_spool is None:
        args.slack_spool = args.out_dir / "slack"
    elif not args.slack_spool.is_absolute():
        args.slack_spool = (Path.cwd() / args.slack_spool).resolve()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    toy_acai_core = load_core(repo_root, args.module_dir)
    obs_dim = observation_dim(toy_acai_core)
    device = torch.device(args.device)
    config = PPOConfig(rollout_steps=args.rollout_steps, batch_size=args.batch_size)
    trainer = PPOTrainer(obs_dim, config, device)
    buffer = RolloutBuffer()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_env = ToyAcaiPPOEnv(toy_acai_core, max_steps=args.steps)
    latest_observations = train_env.reset()

    for episode in range(1, args.episodes + 1):
        observations, reward, info = run_episode(train_env, trainer, buffer=buffer, deterministic=False)
        latest_observations = observations
        metrics = {
            "episode": episode,
            "reward": reward,
            "blue_alive": info.get("blue_alive", 0.0),
            "red_alive": info.get("red_alive", 0.0),
            "outcome": info.get("outcome", 0.0),
            "buffer_steps": len(buffer),
        }
        write_jsonl(args.out_dir / "train_metrics.jsonl", metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

        if len(buffer) >= config.rollout_steps:
            last_values = trainer.values(latest_observations)
            update_stats = trainer.update(buffer, last_values)
            buffer.clear()
            write_jsonl(args.out_dir / "update_metrics.jsonl", {"episode": episode, **update_stats})

        if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0:
            trainer.save(args.out_dir / "checkpoints" / f"ppo_{episode:06d}.pt", {"episode": episode, "obs_dim": obs_dim})

        if args.render_every > 0 and episode % args.render_every == 0:
            eval_metrics = evaluate(toy_acai_core, trainer, args, episode, repo_root)
            print(json.dumps({"eval": eval_metrics}, sort_keys=True), flush=True)

    if len(buffer) > 0:
        last_values = trainer.values(latest_observations)
        update_stats = trainer.update(buffer, last_values)
        write_jsonl(args.out_dir / "update_metrics.jsonl", {"episode": args.episodes, **update_stats})
    trainer.save(args.out_dir / "checkpoints" / "ppo_latest.pt", {"episode": args.episodes, "obs_dim": obs_dim})


if __name__ == "__main__":
    main()
