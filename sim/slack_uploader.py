#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional


SLACK_API = "https://slack.com/api"


def parse_args():
    parser = argparse.ArgumentParser(description="Upload toy-acai GIF spool records to Slack from a login node.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--spool", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ[key] = value


def slack_request(method: str, token: str, payload: dict) -> dict:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SLACK_API}/{method}",
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {result}")
    return result


def post_message(token: str, channel_id: str, text: str, thread_ts: Optional[str] = None) -> str:
    payload = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    result = slack_request("chat.postMessage", token, payload)
    return result["ts"]


def upload_bytes(url: str, file_path: Path) -> None:
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    data = file_path.read_bytes()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": content_type, "Content-Length": str(len(data))},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()


class SlackThread:
    def __init__(self, spool: Path, token: str, channel_id: str, dry_run: bool):
        self.spool = spool
        self.token = token
        self.channel_id = channel_id
        self.dry_run = dry_run
        self.state_path = spool / "thread_ts"
        self.thread_ts = self._read_thread_ts()

    def _read_thread_ts(self) -> Optional[str]:
        if not self.state_path.exists():
            return None
        thread_ts = self.state_path.read_text(encoding="utf-8").strip()
        return thread_ts or None

    def _write_thread_ts(self, thread_ts: str) -> None:
        self.spool.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        tmp_path.write_text(f"{thread_ts}\n", encoding="utf-8")
        tmp_path.replace(self.state_path)

    def post_root(self, text: str) -> str:
        if self.dry_run:
            self.thread_ts = self.thread_ts or "dry-run-thread"
            print(f"dry-run: would post thread-root to {self.channel_id}: {text}")
            return self.thread_ts

        self.thread_ts = post_message(self.token, self.channel_id, text)
        self._write_thread_ts(self.thread_ts)
        print(f"created Slack thread {self.thread_ts} in {self.channel_id}")
        return self.thread_ts

    def ensure(self) -> Optional[str]:
        if self.thread_ts:
            return self.thread_ts
        return self.post_root("toy-acai PPO training updates")


def upload_record(record_path: Path, token: str, channel_id: str, thread: SlackThread, dry_run: bool) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("type") == "thread_root":
        thread.post_root(record.get("comment", "toy-acai PPO training started"))
        return

    gif_path = Path(record["gif_path"])
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF does not exist: {gif_path}")

    thread_ts = thread.ensure()
    if dry_run:
        print(f"dry-run: would upload {gif_path} to {channel_id} thread {thread_ts}: {record.get('comment', '')}")
        return

    upload_url = slack_request(
        "files.getUploadURLExternal",
        token,
        {"filename": gif_path.name, "length": str(gif_path.stat().st_size)},
    )
    upload_bytes(upload_url["upload_url"], gif_path)

    file_info = {"id": upload_url["file_id"], "title": gif_path.name}
    complete_payload = {
        "files": json.dumps([file_info]),
        "channel_id": channel_id,
        "initial_comment": record.get("comment", ""),
    }
    complete_payload["thread_ts"] = thread_ts
    slack_request("files.completeUploadExternal", token, complete_payload)
    print(f"uploaded {gif_path} to {channel_id} thread {thread_ts}")


def move_record(record_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    record_path.replace(destination_dir / record_path.name)


def process_once(spool: Path, token: str, channel_id: str, thread: SlackThread, dry_run: bool) -> int:
    pending = spool / "pending"
    sent = spool / "sent"
    failed = spool / "failed"
    pending.mkdir(parents=True, exist_ok=True)
    count = 0
    for record_path in sorted(pending.glob("*.json")):
        try:
            upload_record(record_path, token, channel_id, thread, dry_run)
            if not dry_run:
                move_record(record_path, sent)
        except Exception as exc:
            print(f"failed {record_path}: {exc}")
            if not dry_run:
                move_record(record_path, failed)
        count += 1
    return count


def main():
    args = parse_args()
    load_dotenv(args.env_file)
    spool = args.spool
    if spool is None:
        spool = Path(os.environ.get("TOY_ACAI_SLACK_SPOOL", "outputs/rl/default/slack"))
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    if not args.dry_run and (not token or not channel_id):
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required unless --dry-run is used")
    token = token or "dry-run-token"
    channel_id = channel_id or "dry-run-channel"
    thread = SlackThread(spool, token, channel_id, args.dry_run)

    while True:
        processed = process_once(spool, token, channel_id, thread, args.dry_run)
        if args.once:
            print(f"processed {processed} records")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
