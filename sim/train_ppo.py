#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from toy_acai_rl.env import (
    ToyAcaiPPOEnv,
    load_core,
    observation_dim,
)
from toy_acai_rl.ppo import PPOConfig, PPOTrainer, RolloutBuffer


EPISODE_INFO_METRICS = (
    "episode_steps",
    "terminal_score",
    "mean_accel",
    "mean_turn",
    "mean_abs_turn",
    "fire_input_rate",
    "reward_mean",
    "survival_reward",
    "advantage_reward",
    "movement_reward",
    "mean_movement_distance",
    "kill_reward",
    "death_penalty",
    "blue_kills",
    "blue_losses",
    "hit_events",
)


def parse_args():
    # 学習条件はコマンドライン引数で変えられるようにしておく。
    # 例: rollout_steps は「何ステップ分の経験をためてから PPO 更新するか」を表す。
    parser = argparse.ArgumentParser(description="Train a PPO policy for the toy-acai simulator.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rl/default"))
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--module-dir", type=Path, default=None)
    parser.add_argument("--slack-spool", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eval-fire-threshold", type=float, default=0.15)
    parser.add_argument("--fire-bias-init", type=float, default=0.4)
    parser.add_argument("--log-std-init", type=float, default=-0.8)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--random-start-steps", type=int, default=120)
    parser.add_argument("--learner-count", type=int, default=int(os.environ.get("TOY_ACAI_LEARNER_COUNT", "1")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
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
        "comment": (
            f"toy-acai PPO episode {int(metrics['episode'])}: "
            f"reward={metrics['reward']:.3f}, outcome={metrics['outcome']:+.0f}, "
            f"terminal_score={metrics.get('terminal_score', 0.0):.3f}"
        ),
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(final_path)


def make_slack_thread_root_record(spool_root: Path, args, repo_root: Path) -> None:
    pending = spool_root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    record_id = f"000000_thread_root_{int(time.time())}"
    tmp_path = pending / f".{record_id}.tmp"
    final_path = pending / f"{record_id}.json"
    payload = {
        "type": "thread_root",
        "attachment_path": str((repo_root / "docs" / "rl_model_overview.md").resolve()),
        "comment": (
            "toy-acai PPO training started: "
            f"episodes={args.episodes}, steps={args.steps}, render_every={args.render_every}, "
            f"random_start_steps={args.random_start_steps}"
        ),
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(final_path)


def choose_hidden_dim(args) -> int:
    # 新規学習では現在の標準サイズを使い、再開時はチェックポイント側のモデル形状を優先する。
    # 形状が違うモデルに重みを読み込むと失敗するため、hidden_dim は特に合わせる必要がある。
    if args.hidden_dim is not None:
        return int(args.hidden_dim)
    if args.resume_checkpoint is None:
        return 256
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})
    if "hidden_dim" in checkpoint_config:
        return int(checkpoint_config["hidden_dim"])
    return 128


def run_episode(
    env: ToyAcaiPPOEnv,
    trainer: PPOTrainer,
    buffer: Optional[RolloutBuffer],
    deterministic: bool = False,
    value_gif: Optional[object] = None,
):
    # 1 エピソード分だけ環境を動かす。
    # 学習時は buffer に経験を保存し、評価時は buffer=None にして方策を更新しない。
    observations = env.reset()
    if value_gif is not None:
        env.take_render_frame()
    total_reward = 0.0
    final_info = {}
    action_count = 0
    accel_sum = 0.0
    turn_sum = 0.0
    abs_turn_sum = 0.0
    fire_sum = 0.0
    episode_steps = 0
    for _ in range(env.max_steps):
        # trainer.act は各味方機の観測から行動を決める。
        # raw_actions は PPO の確率計算用、env_actions は環境へ渡せる範囲に整形済みの行動。
        raw_actions, env_actions, log_probs, values = trainer.act(observations, deterministic=deterministic)
        action_count += int(env_actions.shape[0])
        accel_sum += float(np.sum(env_actions[:, 0]))
        turn_sum += float(np.sum(env_actions[:, 1]))
        abs_turn_sum += float(np.sum(np.abs(env_actions[:, 1])))
        fire_sum += float(np.sum(env_actions[:, 2]))
        result = env.step(env_actions)
        if value_gif is not None:
            frame = env.take_render_frame()
            if frame is not None:
                value_gif.record(frame, values)
        if buffer is not None:
            # PPO では「当時の行動確率」と「価値推定」を後で使うため、
            # 観測・行動・報酬だけでなく log_prob と value も一緒に保存する。
            buffer.add(observations, raw_actions, log_probs, result.rewards, result.done, values)
        total_reward += float(np.mean(result.rewards))
        observations = result.observations
        final_info = result.info
        episode_steps += 1
        if result.done:
            break
    if action_count > 0:
        final_info["mean_accel"] = accel_sum / action_count
        final_info["mean_turn"] = turn_sum / action_count
        final_info["mean_abs_turn"] = abs_turn_sum / action_count
        final_info["fire_input_rate"] = fire_sum / action_count
    final_info["episode_steps"] = float(episode_steps)
    return observations, total_reward, final_info


def add_episode_info_metrics(metrics: dict, info: dict) -> None:
    for key in EPISODE_INFO_METRICS:
        metrics[key] = float(info.get(key, 0.0))


def evaluate(toy_acai_core, trainer: PPOTrainer, args, episode: int, repo_root: Path):
    # 評価では決定論的に動かす。学習中の探索ノイズを切ることで、
    # その時点の方策がどれくらい安定して勝てるかを見やすくする。
    from toy_acai_rl.value_gif import ValueGifRecorder

    media_dir = args.out_dir / "media"
    gif_path = media_dir / f"episode_{episode:06d}.gif"
    module_dir = args.module_dir.resolve() if args.module_dir is not None else repo_root / "linux-python" / "build"
    original_cwd = Path.cwd()
    env = ToyAcaiPPOEnv(
        toy_acai_core,
        max_steps=args.steps,
        render=True,
        module_dir=module_dir,
        random_start_steps=args.random_start_steps,
        learner_count=args.learner_count,
        rng=np.random.default_rng(args.seed + episode),
    )
    value_gif = ValueGifRecorder(gif_path, render_interval=env.render_interval)
    try:
        _, reward, info = run_episode(
            env,
            trainer,
            buffer=None,
            deterministic=True,
            value_gif=value_gif,
        )
    finally:
        os.chdir(original_cwd)
    value_gif.save()

    metrics = {
        "episode": episode,
        "reward": reward,
        "blue_alive": info.get("blue_alive", 0.0),
        "red_alive": info.get("red_alive", 0.0),
        "outcome": info.get("outcome", 0.0),
        "gif": str(gif_path),
    }
    add_episode_info_metrics(metrics, info)
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
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    if args.slack_spool is None:
        args.slack_spool = args.out_dir / "slack"
    elif not args.slack_spool.is_absolute():
        args.slack_spool = (Path.cwd() / args.slack_spool).resolve()
    if args.slack_spool is not None:
        make_slack_thread_root_record(args.slack_spool, args, repo_root)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    toy_acai_core = load_core(repo_root, args.module_dir)
    # 環境から得られる観測ベクトルの長さを調べ、ニューラルネットの入力次元にする。
    obs_dim = observation_dim(toy_acai_core)
    device = torch.device(args.device)
    hidden_dim = choose_hidden_dim(args)
    config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=args.lr,
        update_epochs=args.update_epochs,
        rollout_steps=args.rollout_steps,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
        fire_bias_init=args.fire_bias_init,
        eval_fire_threshold=args.eval_fire_threshold,
        log_std_init=args.log_std_init,
        hidden_dim=hidden_dim,
    )
    agent_count = int(np.clip(args.learner_count, 1, int(toy_acai_core.TEAM_FIGHTER_COUNT)))
    args.learner_count = agent_count
    trainer = PPOTrainer(
        obs_dim,
        config,
        device,
        agent_count=agent_count,
    )
    start_episode = 1
    if args.resume_checkpoint is not None:
        # 途中再開時は重みだけでなく、観測次元が現在の環境と一致するかも確認する。
        # 観測設計を変えたチェックポイントをそのまま使うと、入力サイズが合わない。
        checkpoint = trainer.load(args.resume_checkpoint)
        checkpoint_obs_dim = int(checkpoint.get("obs_dim", obs_dim))
        if checkpoint_obs_dim != obs_dim:
            raise ValueError(
                f"checkpoint obs_dim={checkpoint_obs_dim} does not match env obs_dim={obs_dim}"
            )
        start_episode = int(checkpoint.get("episode", 0)) + 1
    buffer = RolloutBuffer(agent_count=agent_count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 学習用環境は 1 つを使い回す。各エピソードの先頭で reset される。
    train_env = ToyAcaiPPOEnv(
        toy_acai_core,
        max_steps=args.steps,
        random_start_steps=args.random_start_steps,
        learner_count=args.learner_count,
        rng=np.random.default_rng(args.seed),
    )
    latest_observations = train_env.reset()

    for episode in range(start_episode, args.episodes + 1):
        # ここが学習の基本サイクル:
        # 1. 方策で 1 エピソード動く
        # 2. 経験を buffer にためる
        # 3. 一定量たまったら PPO で方策を更新する
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
        add_episode_info_metrics(metrics, info)
        write_jsonl(args.out_dir / "train_metrics.jsonl", metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)

        if len(buffer) >= config.rollout_steps:
            # エピソードが途中で終わっていない最後の状態には、将来価値を推定して足し込む。
            # これを bootstrap と呼び、GAE/return の計算に使う。
            last_values = trainer.values(latest_observations)
            update_stats = trainer.update(buffer, last_values)
            buffer.clear()
            write_jsonl(args.out_dir / "update_metrics.jsonl", {"episode": episode, **update_stats})

        if args.checkpoint_every > 0 and episode % args.checkpoint_every == 0:
            checkpoint_extra = {"episode": episode, "obs_dim": obs_dim, "learner_count": agent_count}
            trainer.save(args.out_dir / "checkpoints" / f"ppo_{episode:06d}.pt", checkpoint_extra)
            trainer.save(args.out_dir / "checkpoints" / "ppo_latest.pt", checkpoint_extra)

        if args.render_every > 0 and episode % args.render_every == 0:
            # GIF 生成つきの評価は重いので、毎エピソードではなく間隔を空けて実行する。
            eval_metrics = evaluate(toy_acai_core, trainer, args, episode, repo_root)
            print(json.dumps({"eval": eval_metrics}, sort_keys=True), flush=True)

    if len(buffer) > 0:
        # 最後に rollout_steps 未満の経験が余っていても、捨てずに一度だけ更新する。
        last_values = trainer.values(latest_observations)
        update_stats = trainer.update(buffer, last_values)
        write_jsonl(args.out_dir / "update_metrics.jsonl", {"episode": args.episodes, **update_stats})
    trainer.save(args.out_dir / "checkpoints" / "ppo_latest.pt", {"episode": args.episodes, "obs_dim": obs_dim})


if __name__ == "__main__":
    main()
