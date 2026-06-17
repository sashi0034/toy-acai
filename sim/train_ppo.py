#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from toy_acai_rl.env import (
    ToyAcaiPPOEnv,
    load_core,
    observation_dim,
)
from toy_acai_rl.ppo import PPOConfig, PPOTrainer, RolloutBuffer
from toy_acai_rl.curriculum import (
    CURRICULUM_EVAL_EVERY,
    CURRICULUM_LEARNER_COUNT,
    CURRICULUM_PROMOTION_EVALS,
    CURRICULUM_PROMOTION_WINS,
    CURRICULUM_STAGE_MAX_EPISODES,
    CURRICULUM_STAGES,
    should_promote_curriculum_stage,
)


SLACK_REWARD_CHART_POST_INTERVAL = 10

EPISODE_INFO_METRICS = (
    "episode_steps",
    "terminal_score",
    "mean_accel",
    "mean_turn",
    "mean_abs_turn",
    "fire_input_rate",
    "reward_mean",
    "out_of_bounds_penalty",
    "evasion_reward",
    "movement_reward",
    "mean_movement_distance",
    "missile_fire_reward",
    "kill_reward",
    "death_penalty",
    "blue_kills",
    "blue_losses",
    "hit_events",
)

# auxiliary_agent_rewards が step ごとに返す「その step だけの増分」を、
# エピソード単位の累計に置き換えたいキー。
# 最終 step の値だけを残すと、終了直前にたまたま発生していなかった成分は
# 0 として記録されてしまい「ミサイル回避報酬が動いていない」ように見えてしまう。
EPISODE_CUMULATIVE_INFO_KEYS = (
    "survival_reward",
    "out_of_bounds_penalty",
    "evasion_reward",
    "movement_reward",
    "missile_fire_reward",
    "kill_reward",
    "death_penalty",
    "blue_kills",
    "blue_losses",
    "hit_events",
)


class EpisodeInfoAggregator:
    """1 エピソード分の補助報酬・カウンタを step ごとに集計するヘルパー。"""

    def __init__(self):
        self._totals = {key: 0.0 for key in EPISODE_CUMULATIVE_INFO_KEYS}
        self._mean_movement_sum = 0.0
        self._mean_movement_steps = 0

    def add(self, info: dict) -> None:
        for key in EPISODE_CUMULATIVE_INFO_KEYS:
            self._totals[key] += float(info.get(key, 0.0))
        self._mean_movement_sum += float(info.get("mean_movement_distance", 0.0))
        self._mean_movement_steps += 1

    def apply(self, info: dict) -> dict:
        # 最終 step の info(終端スコアや勝敗のように「最後の値」が欲しいキー)を
        # ベースに、累積したいキーだけ上書きする。
        merged = dict(info)
        merged.update(self._totals)
        if self._mean_movement_steps > 0:
            merged["mean_movement_distance"] = (
                self._mean_movement_sum / self._mean_movement_steps
            )
        else:
            merged["mean_movement_distance"] = 0.0
        return merged


def parse_args():
    # 学習条件はコマンドライン引数で変えられるようにしておく。
    # 例: rollout_steps は「何ステップ分の経験をためてから PPO 更新するか」を表す。
    parser = argparse.ArgumentParser(description="Train a PPO policy for the toy-acai simulator.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=1200)
    # --out-dir は学習結果の親ディレクトリで、実際の出力は内部に作る run_<timestamp> 以下になる。
    # こうすることで、再実行しても過去の checkpoint/メトリクスを上書きしない。
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rl"))
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--module-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--eval-fire-threshold", type=float, default=0.15)
    parser.add_argument("--fire-bias-init", type=float, default=1.0)
    parser.add_argument("--log-std-init", type=float, default=-0.8)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--learner-count", type=int, default=int(os.environ.get("TOY_ACAI_LEARNER_COUNT", "1")))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args()


def write_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def make_run_dir(base_dir: Path) -> Path:
    # 学習の起動ごとにユニークなディレクトリを切る。
    # 1 秒以内に複数ジョブが立ち上がっても衝突しないように、必要なら連番でずらす。
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    job_id = os.environ.get("SLURM_JOB_ID")
    name = f"run_{timestamp}_{job_id}" if job_id else f"run_{timestamp}"
    candidate = base_dir / name
    counter = 1
    while candidate.exists():
        candidate = base_dir / f"{name}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def update_latest_symlink(base_dir: Path, run_dir: Path) -> None:
    # base_dir/latest を最新の run_dir に向け直す。
    # slack uploader など外部プロセスはこの latest を参照する想定。
    link_path = base_dir / "latest"
    target = run_dir.name
    tmp_link = base_dir / f".latest.{os.getpid()}.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(target, tmp_link)
    os.replace(tmp_link, link_path)


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


def reward_history_from_jsonl(
    path: Path,
    max_episode: int,
    opponent_count: Optional[int] = None,
    curriculum_stage: Optional[int] = None,
) -> list[dict]:
    if max_episode <= 0 or not path.exists():
        return []
    history = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            episode = int(row["episode"])
            if episode > max_episode:
                continue
            row_opponent_count = row.get("opponent_count")
            row_curriculum_stage = row.get("curriculum_stage")
            if opponent_count is not None and int(row_opponent_count) != int(opponent_count):
                continue
            if curriculum_stage is not None and int(row_curriculum_stage) != int(curriculum_stage):
                continue
            history.append(
                {
                    "episode": episode,
                    "reward": float(row["reward"]),
                    "opponent_count": int(row_opponent_count) if row_opponent_count is not None else None,
                    "curriculum_stage": int(row_curriculum_stage) if row_curriculum_stage is not None else None,
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return history


def _format_chart_tick(value: float) -> str:
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def make_reward_chart_image(path: Path, reward_history: list[dict], stage_label: Optional[str] = None) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not reward_history:
        raise ValueError("reward_history must contain at least one point")

    points = sorted(
        (int(row["episode"]), float(row["reward"]))
        for row in reward_history
    )
    episodes = [episode for episode, _reward in points]
    rewards = [reward for _episode, reward in points]

    width = 900
    height = 520
    margin_left = 86
    margin_right = 34
    margin_top = 52
    margin_bottom = 76
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    x_min = min(episodes)
    x_max = max(episodes)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    y_min = min(rewards)
    y_max = max(rewards)
    if y_min == y_max:
        padding = max(1.0, abs(y_min) * 0.1)
    else:
        padding = (y_max - y_min) * 0.12
    y_min -= padding
    y_max += padding

    def map_x(episode: int) -> float:
        return plot_left + (episode - x_min) / (x_max - x_min) * plot_width

    def map_y(reward: float) -> float:
        return plot_bottom - (reward - y_min) / (y_max - y_min) * plot_height

    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), fill=(255, 255, 255), outline=(202, 208, 219))

    y_tick_count = 5
    for index in range(y_tick_count + 1):
        ratio = index / y_tick_count
        value = y_min + (y_max - y_min) * ratio
        y = plot_bottom - ratio * plot_height
        draw.line((plot_left, y, plot_right, y), fill=(226, 231, 238))
        label = _format_chart_tick(value)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((plot_left - 10 - (bbox[2] - bbox[0]), y - 6), label, fill=(66, 78, 96), font=font)

    x_tick_count = min(6, max(1, len(points) - 1))
    for index in range(x_tick_count + 1):
        ratio = index / x_tick_count if x_tick_count else 0.0
        value = int(round(x_min + (x_max - x_min) * ratio))
        x = map_x(value)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill=(110, 122, 142))
        label = str(value)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 12), label, fill=(66, 78, 96), font=font)

    mapped_points = [(map_x(episode), map_y(reward)) for episode, reward in points]
    if len(mapped_points) >= 2:
        draw.line(mapped_points, fill=(30, 103, 210), width=3)
    for x, y in mapped_points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(239, 125, 47), outline=(255, 255, 255), width=2)

    latest_episode, latest_reward = points[-1]
    stage_suffix = f" - {stage_label}" if stage_label else ""
    title = f"toy-acai PPO reward trend{stage_suffix} ({len(points)} evals)"
    subtitle = f"latest: episode {latest_episode}, reward={latest_reward:.3f}"
    draw.text((plot_left, 18), title, fill=(28, 38, 54), font=title_font)
    draw.text((plot_left + 330, 20), subtitle, fill=(66, 78, 96), font=font)

    x_label = "episode"
    x_bbox = draw.textbbox((0, 0), x_label, font=font)
    draw.text(((plot_left + plot_right - (x_bbox[2] - x_bbox[0])) / 2, height - 30), x_label, fill=(28, 38, 54), font=font)
    y_label = "reward"
    draw.text((18, plot_top + plot_height / 2 - 6), y_label, fill=(28, 38, 54), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def make_reward_chart_spool_record(spool_root: Path, chart_path: Path, reward_history: list[dict]) -> None:
    pending = spool_root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    latest = reward_history[-1]
    opponent_count = latest.get("opponent_count")
    stage_text = f", red={int(opponent_count)}" if opponent_count is not None else ""
    record_id = f"reward_chart_{int(latest['episode']):06d}_{int(time.time())}"
    tmp_path = pending / f".{record_id}.tmp"
    final_path = pending / f"{record_id}.json"
    payload = {
        "file_path": str(chart_path.resolve()),
        "episode": int(latest["episode"]),
        "reward": float(latest["reward"]),
        "comment": (
            f"toy-acai PPO reward trend after {SLACK_REWARD_CHART_POST_INTERVAL} Slack updates: "
            f"latest episode {int(latest['episode'])}{stage_text}, reward={float(latest['reward']):.3f}"
        ),
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(final_path)


def reset_reward_chart_stage(reward_history: list[dict]) -> int:
    reward_history.clear()
    return 0


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
            "random_start_positions=True"
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
    trainer.reset_exploration_noise()
    if value_gif is not None:
        initial_frame = env.take_render_frame()
        if initial_frame is not None:
            # 短期決着でも GIF が 0 フレームにならないよう、開始直後フレームを先に記録する。
            initial_agent_count = int(np.asarray(observations).shape[0])
            initial_values = np.zeros((initial_agent_count,), dtype=np.float32)
            initial_rewards = np.zeros((initial_agent_count,), dtype=np.float32)
            value_gif.record(initial_frame, initial_values, initial_rewards)
    total_reward = 0.0
    final_info = {}
    action_count = 0
    accel_sum = 0.0
    turn_sum = 0.0
    abs_turn_sum = 0.0
    fire_sum = 0.0
    episode_steps = 0
    # GIF オーバーレイの reward 表示はそのフレーム時点の累計報酬にする。
    cumulative_agent_rewards: Optional[np.ndarray] = None
    aggregator = EpisodeInfoAggregator()
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
        step_rewards = np.asarray(result.rewards, dtype=np.float32)
        if cumulative_agent_rewards is None:
            cumulative_agent_rewards = step_rewards.copy()
        else:
            cumulative_agent_rewards = cumulative_agent_rewards + step_rewards
        if value_gif is not None:
            frame = env.take_render_frame()
            if frame is not None:
                value_gif.record(frame, values, cumulative_agent_rewards)
        if buffer is not None:
            # PPO では「当時の行動確率」と「価値推定」を後で使うため、
            # 観測・行動・報酬だけでなく log_prob と value も一緒に保存する。
            buffer.add(observations, raw_actions, log_probs, result.rewards, result.done, values)
        total_reward += float(np.mean(result.rewards))
        observations = result.observations
        # info 内の補助報酬は step 毎の増分なので、最後の step だけ残すと
        # ほぼ常に 0 に見えてしまう。ここでエピソード全体の累計を別に持つ。
        aggregator.add(result.info)
        final_info = result.info
        episode_steps += 1
        if result.done:
            break
    final_info = aggregator.apply(final_info)
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


def make_checkpoint_extra(
    *,
    episode: int,
    obs_dim: int,
    learner_count: int,
    stage_index: int,
    stage_episode: int,
) -> dict:
    opponent_count = CURRICULUM_STAGES[stage_index]
    return {
        "episode": episode,
        "global_episode": episode,
        "obs_dim": obs_dim,
        "learner_count": learner_count,
        "curriculum_stage": stage_index + 1,
        "opponent_count": opponent_count,
        "stage_episode": stage_episode,
    }


def curriculum_stage_from_checkpoint(checkpoint: dict) -> Tuple[int, int]:
    opponent_count = int(checkpoint.get("opponent_count", CURRICULUM_STAGES[0]))
    if opponent_count in CURRICULUM_STAGES:
        stage_index = CURRICULUM_STAGES.index(opponent_count)
    else:
        stage_index = int(checkpoint.get("curriculum_stage", 1)) - 1
        stage_index = int(np.clip(stage_index, 0, len(CURRICULUM_STAGES) - 1))
    stage_episode = int(checkpoint.get("stage_episode", 0))
    return stage_index, stage_episode


def make_train_env(toy_acai_core, args, opponent_count: int) -> ToyAcaiPPOEnv:
    return ToyAcaiPPOEnv(
        toy_acai_core,
        max_steps=args.steps,
        random_start_positions=True,
        learner_count=args.learner_count,
        opponent_count=opponent_count,
        rng=np.random.default_rng(args.seed),
    )


def evaluate_curriculum(
    toy_acai_core,
    trainer: PPOTrainer,
    args,
    episode: int,
    stage_index: int,
    stage_episode: int,
) -> dict:
    opponent_count = CURRICULUM_STAGES[stage_index]
    wins = 0
    rewards = []
    for eval_index in range(CURRICULUM_PROMOTION_EVALS):
        env = ToyAcaiPPOEnv(
            toy_acai_core,
            max_steps=args.steps,
            random_start_positions=True,
            learner_count=args.learner_count,
            opponent_count=opponent_count,
            rng=np.random.default_rng(args.seed + episode * 1000 + eval_index),
        )
        _, reward, info = run_episode(
            env,
            trainer,
            buffer=None,
            deterministic=True,
        )
        rewards.append(float(reward))
        if float(info.get("outcome", 0.0)) > 0.0:
            wins += 1

    win_rate = wins / float(CURRICULUM_PROMOTION_EVALS)
    promote, reason = should_promote_curriculum_stage(
        stage_index=stage_index,
        stage_episode=stage_episode,
        wins=wins,
        evals=CURRICULUM_PROMOTION_EVALS,
    )
    metrics = {
        "episode": episode,
        "curriculum_stage": stage_index + 1,
        "stage_episode": stage_episode,
        "opponent_count": opponent_count,
        "evals": CURRICULUM_PROMOTION_EVALS,
        "wins": wins,
        "win_rate": win_rate,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "promote": bool(promote),
        "promotion_reason": reason,
    }
    write_jsonl(args.out_dir / "curriculum_metrics.jsonl", metrics)
    return metrics


def promote_curriculum_stage(
    *,
    stage_index: int,
    episode: int,
    stage_episode: int,
    reason: str,
    args,
) -> Tuple[int, int]:
    next_stage_index = min(stage_index + 1, len(CURRICULUM_STAGES) - 1)
    metrics = {
        "episode": episode,
        "from_stage": stage_index + 1,
        "to_stage": next_stage_index + 1,
        "from_opponent_count": CURRICULUM_STAGES[stage_index],
        "to_opponent_count": CURRICULUM_STAGES[next_stage_index],
        "stage_episode": stage_episode,
        "reason": reason,
    }
    write_jsonl(args.out_dir / "curriculum_events.jsonl", metrics)
    print(json.dumps({"curriculum_promotion": metrics}, sort_keys=True), flush=True)
    return next_stage_index, 0


def evaluate(
    toy_acai_core,
    trainer: PPOTrainer,
    args,
    episode: int,
    repo_root: Path,
    stage_index: int,
    stage_episode: int,
):
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
        random_start_positions=True,
        learner_count=args.learner_count,
        opponent_count=CURRICULUM_STAGES[stage_index],
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
    gif_saved = len(value_gif.frames) > 0
    if gif_saved:
        value_gif.save()
    else:
        print(
            json.dumps(
                {
                    "eval_gif_skipped": {
                        "episode": episode,
                        "reason": "no_frames_recorded",
                        "gif": str(gif_path),
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metrics = {
        "episode": episode,
        "reward": reward,
        "blue_alive": info.get("blue_alive", 0.0),
        "red_alive": info.get("red_alive", 0.0),
        "outcome": info.get("outcome", 0.0),
        "curriculum_stage": stage_index + 1,
        "stage_episode": stage_episode,
        "opponent_count": CURRICULUM_STAGES[stage_index],
        "gif": str(gif_path) if gif_saved else "",
        "gif_saved": bool(gif_saved),
    }
    add_episode_info_metrics(metrics, info)
    write_jsonl(args.out_dir / "eval_metrics.jsonl", metrics)
    if gif_saved:
        make_spool_record(args.slack_spool, gif_path, metrics)
    return metrics


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_out_dir = args.out_dir.resolve()
    args.out_dir = make_run_dir(base_out_dir)
    update_latest_symlink(base_out_dir, args.out_dir)
    print(json.dumps({"run_dir": str(args.out_dir)}, sort_keys=True), flush=True)
    if args.module_dir is not None:
        args.module_dir = args.module_dir.resolve()
    if args.resume_checkpoint is not None:
        args.resume_checkpoint = args.resume_checkpoint.resolve()
    args.slack_spool = args.out_dir / "slack"
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
    args.learner_count = CURRICULUM_LEARNER_COUNT
    agent_count = int(np.clip(args.learner_count, 1, int(toy_acai_core.TEAM_FIGHTER_COUNT)))
    args.learner_count = agent_count
    trainer = PPOTrainer(
        obs_dim,
        config,
        device,
        agent_count=agent_count,
    )
    start_episode = 1
    resumed_episode = 0
    stage_index = 0
    stage_episode = 0
    if args.resume_checkpoint is not None:
        # 途中再開時は重みだけでなく、観測次元が現在の環境と一致するかも確認する。
        # 観測設計を変えたチェックポイントをそのまま使うと、入力サイズが合わない。
        checkpoint = trainer.load(args.resume_checkpoint)
        checkpoint_obs_dim = int(checkpoint.get("obs_dim", obs_dim))
        if checkpoint_obs_dim != obs_dim:
            raise ValueError(
                f"checkpoint obs_dim={checkpoint_obs_dim} does not match env obs_dim={obs_dim}"
            )
        resumed_episode = int(checkpoint.get("global_episode", checkpoint.get("episode", 0)))
        start_episode = resumed_episode + 1
        stage_index, stage_episode = curriculum_stage_from_checkpoint(checkpoint)
    buffer = RolloutBuffer(agent_count=agent_count)
    eval_reward_history = reward_history_from_jsonl(
        args.out_dir / "eval_metrics.jsonl",
        resumed_episode,
        opponent_count=CURRICULUM_STAGES[stage_index],
    )
    slack_update_post_count = len(eval_reward_history)

    # 学習用環境は 1 つを使い回す。各エピソードの先頭で reset される。
    train_env = make_train_env(toy_acai_core, args, CURRICULUM_STAGES[stage_index])
    latest_observations = train_env.reset()

    for episode in range(start_episode, args.episodes + 1):
        # ここが学習の基本サイクル:
        # 1. 方策で 1 エピソード動く
        # 2. 経験を buffer にためる
        # 3. 一定量たまったら PPO で方策を更新する
        stage_episode += 1
        observations, reward, info = run_episode(train_env, trainer, buffer=buffer, deterministic=False)
        latest_observations = observations
        metrics = {
            "episode": episode,
            "reward": reward,
            "blue_alive": info.get("blue_alive", 0.0),
            "red_alive": info.get("red_alive", 0.0),
            "outcome": info.get("outcome", 0.0),
            "buffer_steps": len(buffer),
            "curriculum_stage": stage_index + 1,
            "stage_episode": stage_episode,
            "opponent_count": CURRICULUM_STAGES[stage_index],
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
            checkpoint_extra = make_checkpoint_extra(
                episode=episode,
                obs_dim=obs_dim,
                learner_count=agent_count,
                stage_index=stage_index,
                stage_episode=stage_episode,
            )
            trainer.save(args.out_dir / "checkpoints" / f"ppo_{episode:06d}.pt", checkpoint_extra)
            trainer.save(args.out_dir / "checkpoints" / "ppo_latest.pt", checkpoint_extra)
            trainer.save(
                args.out_dir / "checkpoints" / f"ppo_stage_red{CURRICULUM_STAGES[stage_index]}_latest.pt",
                checkpoint_extra,
            )

        if args.render_every > 0 and episode % args.render_every == 0:
            # GIF 生成つきの評価は重いので、毎エピソードではなく間隔を空けて実行する。
            eval_metrics = evaluate(
                toy_acai_core,
                trainer,
                args,
                episode,
                repo_root,
                stage_index,
                stage_episode,
            )
            eval_reward_history.append(
                {
                    "episode": int(eval_metrics["episode"]),
                    "reward": float(eval_metrics["reward"]),
                    "opponent_count": int(eval_metrics["opponent_count"]),
                    "curriculum_stage": int(eval_metrics["curriculum_stage"]),
                }
            )
            slack_update_post_count += 1
            if slack_update_post_count % SLACK_REWARD_CHART_POST_INTERVAL == 0:
                chart_path = args.out_dir / "media" / f"reward_chart_{episode:06d}.png"
                make_reward_chart_image(
                    chart_path,
                    eval_reward_history,
                    stage_label=f"red={CURRICULUM_STAGES[stage_index]}",
                )
                make_reward_chart_spool_record(args.slack_spool, chart_path, eval_reward_history)
            print(json.dumps({"eval": eval_metrics}, sort_keys=True), flush=True)

        if stage_index < len(CURRICULUM_STAGES) - 1:
            curriculum_metrics = None
            if episode % CURRICULUM_EVAL_EVERY == 0:
                curriculum_metrics = evaluate_curriculum(
                    toy_acai_core,
                    trainer,
                    args,
                    episode,
                    stage_index,
                    stage_episode,
                )
                print(json.dumps({"curriculum_eval": curriculum_metrics}, sort_keys=True), flush=True)
                promote = bool(curriculum_metrics["promote"])
                promotion_reason = str(curriculum_metrics["promotion_reason"])
            else:
                promote, promotion_reason = should_promote_curriculum_stage(
                    stage_index=stage_index,
                    stage_episode=stage_episode,
                    wins=0,
                    evals=0,
                )

            if promote:
                stage_index, stage_episode = promote_curriculum_stage(
                    stage_index=stage_index,
                    episode=episode,
                    stage_episode=stage_episode,
                    reason=promotion_reason,
                    args=args,
                )
                buffer.clear()
                slack_update_post_count = reset_reward_chart_stage(eval_reward_history)
                train_env = make_train_env(toy_acai_core, args, CURRICULUM_STAGES[stage_index])
                latest_observations = train_env.reset()

    if len(buffer) > 0:
        # 最後に rollout_steps 未満の経験が余っていても、捨てずに一度だけ更新する。
        last_values = trainer.values(latest_observations)
        update_stats = trainer.update(buffer, last_values)
        write_jsonl(args.out_dir / "update_metrics.jsonl", {"episode": args.episodes, **update_stats})
    trainer.save(
        args.out_dir / "checkpoints" / "ppo_latest.pt",
        make_checkpoint_extra(
            episode=args.episodes,
            obs_dim=obs_dim,
            learner_count=agent_count,
            stage_index=stage_index,
            stage_episode=stage_episode,
        ),
    )


if __name__ == "__main__":
    main()
