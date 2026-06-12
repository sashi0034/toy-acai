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


def upload_record(record_path: Path, token: str, channel_id: str, thread_ts: Optional[str], dry_run: bool) -> None:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    gif_path = Path(record["gif_path"])
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF does not exist: {gif_path}")

    if dry_run:
        print(f"dry-run: would upload {gif_path} to {channel_id}: {record.get('comment', '')}")
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
    if thread_ts:
        complete_payload["thread_ts"] = thread_ts
    slack_request("files.completeUploadExternal", token, complete_payload)
    print(f"uploaded {gif_path} to {channel_id}")


def move_record(record_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    record_path.replace(destination_dir / record_path.name)


def process_once(spool: Path, token: str, channel_id: str, thread_ts: Optional[str], dry_run: bool) -> int:
    pending = spool / "pending"
    sent = spool / "sent"
    failed = spool / "failed"
    pending.mkdir(parents=True, exist_ok=True)
    count = 0
    for record_path in sorted(pending.glob("*.json")):
        try:
            upload_record(record_path, token, channel_id, thread_ts, dry_run)
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
    thread_ts = os.environ.get("SLACK_THREAD_TS")
    if not args.dry_run and (not token or not channel_id):
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required unless --dry-run is used")
    token = token or "dry-run-token"
    channel_id = channel_id or "dry-run-channel"

    while True:
        processed = process_once(spool, token, channel_id, thread_ts, args.dry_run)
        if args.once:
            print(f"processed {processed} records")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
