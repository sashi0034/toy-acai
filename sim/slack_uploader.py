"""Upload Slack spool records from a network-enabled node."""

import argparse
import json
import os
from pathlib import Path
import time

from ._slack import (
    SlackThread,
    file_share_ts,
    load_dotenv,
    training_started_message,
    upload_file,
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload toy-acai sim Slack spool records.")
    parser.add_argument("--env-file", type=Path, default=repository_root() / ".env")
    parser.add_argument("--spool", type=Path)
    parser.add_argument("--spool-root", type=Path, default=repository_root() / "outputs")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def discover_spools(spool_root: Path) -> list[Path]:
    if not spool_root.exists():
        return []
    return sorted(
        path / "slack"
        for path in spool_root.iterdir()
        if path.is_dir() and not path.is_symlink() and (path / "slack").is_dir()
    )


def move_record(record_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    record_path.replace(destination / record_path.name)


def upload_record(record_path: Path, thread: SlackThread) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("type") == "thread_root":
        attachment = Path(record["attachment_path"])
        comment = record.get("comment") or training_started_message()
        if thread.dry_run:
            thread.post_root(comment, attachment)
            return
        file_id = record.get("uploaded_file_id")
        if not file_id:
            file_id = upload_file(
                thread.token,
                thread.channel_id,
                attachment,
                comment,
            )
            record["uploaded_file_id"] = file_id
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        thread.thread_ts = file_share_ts(thread.token, str(file_id), thread.channel_id)
        thread._write_thread_ts(thread.thread_ts)
        return

    file_path = Path(record["file_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Slack attachment does not exist: {file_path}")
    if thread.dry_run:
        print(f"dry-run: would upload {file_path} to thread {thread.ensure()}")
        return
    upload_file(
        thread.token,
        thread.channel_id,
        file_path,
        record.get("comment", ""),
        thread.ensure(),
    )


def process_once(spool: Path, thread: SlackThread) -> int:
    pending = spool / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    processed = 0
    for record_path in sorted(pending.glob("*.json")):
        try:
            upload_record(record_path, thread)
            if not thread.dry_run:
                move_record(record_path, spool / "sent")
        except Exception as error:
            print(f"failed {record_path}: {error}")
            if not thread.dry_run:
                move_record(record_path, spool / "failed")
        processed += 1
    return processed


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    if not args.dry_run and (not token or not channel_id):
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required unless --dry-run is used")
    token = token or "dry-run-token"
    channel_id = channel_id or "dry-run-channel"
    threads: dict[Path, SlackThread] = {}

    while True:
        spools = [args.spool] if args.spool else discover_spools(args.spool_root)
        count = 0
        for spool in spools:
            resolved = spool.resolve()
            thread = threads.setdefault(
                resolved,
                SlackThread(token, channel_id, spool / "thread_ts", args.dry_run),
            )
            count += process_once(spool, thread)
        if args.once:
            print(f"processed {count} records")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
