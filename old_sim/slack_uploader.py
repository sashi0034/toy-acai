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
from typing import Dict, List, Optional


SLACK_API = "https://slack.com/api"


def parse_args():
    parser = argparse.ArgumentParser(description="Upload toy-acai GIF spool records to Slack from a login node.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--spool", type=Path, default=None)
    parser.add_argument("--spool-root", type=Path, default=Path("outputs/rl"))
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


def upload_file(
    token: str,
    channel_id: str,
    file_path: Path,
    initial_comment: str = "",
    thread_ts: Optional[str] = None,
) -> str:
    upload_url = slack_request(
        "files.getUploadURLExternal",
        token,
        {"filename": file_path.name, "length": str(file_path.stat().st_size)},
    )
    upload_bytes(upload_url["upload_url"], file_path)

    file_info = {"id": upload_url["file_id"], "title": file_path.name}
    complete_payload = {
        "files": json.dumps([file_info]),
        "channel_id": channel_id,
        "initial_comment": initial_comment,
    }
    if thread_ts:
        complete_payload["thread_ts"] = thread_ts
    slack_request("files.completeUploadExternal", token, complete_payload)
    return str(upload_url["file_id"])


def _share_ts_from_entry(share: object) -> Optional[str]:
    if not isinstance(share, dict):
        return None
    ts = share.get("ts") or share.get("thread_ts")
    return str(ts) if ts else None


def _walk_share_entries(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_share_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_share_entries(child)


def _extract_file_share_ts(file_info: dict, channel_id: str) -> Optional[str]:
    shares = file_info.get("shares", {})
    for visibility in ("public", "private"):
        for share in shares.get(visibility, {}).get(channel_id, []):
            ts = _share_ts_from_entry(share)
            if ts:
                return ts
    for share in _walk_share_entries(shares):
        channel = share.get("channel") or share.get("channel_id")
        if channel not in (None, channel_id):
            continue
        ts = _share_ts_from_entry(share)
        if ts:
            return ts
    return None


def file_share_ts(token: str, file_id: str, channel_id: str, attempts: int = 8, delay_seconds: float = 2.0) -> str:
    last_file_info = {}
    for attempt in range(max(1, attempts)):
        result = slack_request("files.info", token, {"file": file_id})
        last_file_info = result.get("file", {})
        ts = _extract_file_share_ts(last_file_info, channel_id)
        if ts:
            return ts
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    share_keys = sorted((last_file_info.get("shares") or {}).keys())
    raise RuntimeError(
        f"could not find Slack share ts for file {file_id} in {channel_id}; "
        f"share groups={share_keys}"
    )


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

    def post_root(self, text: str, attachment_path: Optional[Path] = None) -> str:
        if self.dry_run:
            self.thread_ts = self.thread_ts or "dry-run-thread"
            if attachment_path is not None:
                print(f"dry-run: would post thread-root file {attachment_path} to {self.channel_id}: {text}")
            else:
                print(f"dry-run: would post thread-root to {self.channel_id}: {text}")
            return self.thread_ts

        if attachment_path is None:
            self.thread_ts = post_message(self.token, self.channel_id, text)
        else:
            if not attachment_path.exists():
                raise FileNotFoundError(f"thread root attachment does not exist: {attachment_path}")
            file_id = upload_file(self.token, self.channel_id, attachment_path, initial_comment=text)
            self.thread_ts = file_share_ts(self.token, file_id, self.channel_id)
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
        attachment_path = record.get("attachment_path")
        if attachment_path and not dry_run:
            file_id = record.get("uploaded_file_id")
            if not file_id:
                path = Path(attachment_path)
                if not path.exists():
                    raise FileNotFoundError(f"thread root attachment does not exist: {path}")
                file_id = upload_file(
                    token,
                    channel_id,
                    path,
                    initial_comment=record.get("comment", "toy-acai PPO training started"),
                )
                record["uploaded_file_id"] = file_id
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            thread.thread_ts = file_share_ts(token, str(file_id), channel_id)
            thread._write_thread_ts(thread.thread_ts)
            print(f"created Slack thread {thread.thread_ts} in {channel_id}")
            return
        thread.post_root(
            record.get("comment", "toy-acai PPO training started"),
            Path(attachment_path) if attachment_path else None,
        )
        return

    file_path = Path(record.get("file_path") or record["gif_path"])
    if not file_path.exists():
        raise FileNotFoundError(f"Slack attachment does not exist: {file_path}")

    thread_ts = thread.ensure()
    if dry_run:
        print(f"dry-run: would upload {file_path} to {channel_id} thread {thread_ts}: {record.get('comment', '')}")
        return

    upload_file(token, channel_id, file_path, initial_comment=record.get("comment", ""), thread_ts=thread_ts)
    print(f"uploaded {file_path} to {channel_id} thread {thread_ts}")


def move_record(record_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    record_path.replace(destination_dir / record_path.name)


def discover_spools(spool_root: Path) -> List[Path]:
    if not spool_root.exists():
        return []
    return sorted(
        slack_dir
        for run_dir in spool_root.glob("run_*")
        if run_dir.is_dir() and not run_dir.is_symlink()
        for slack_dir in [run_dir / "slack"]
        if slack_dir.is_dir()
    )


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


def process_all_once(
    spools: List[Path],
    token: str,
    channel_id: str,
    threads: Dict[Path, SlackThread],
    dry_run: bool,
) -> int:
    count = 0
    for spool in spools:
        resolved_spool = spool.resolve()
        thread = threads.get(resolved_spool)
        if thread is None:
            thread = SlackThread(spool, token, channel_id, dry_run)
            threads[resolved_spool] = thread
        count += process_once(spool, token, channel_id, thread, dry_run)
    return count


def main():
    args = parse_args()
    load_dotenv(args.env_file)
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    if not args.dry_run and (not token or not channel_id):
        raise SystemExit("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required unless --dry-run is used")
    token = token or "dry-run-token"
    channel_id = channel_id or "dry-run-channel"
    threads: Dict[Path, SlackThread] = {}

    while True:
        if args.spool is not None:
            spools = [args.spool]
        else:
            # 複数の学習 run が同時に動くと latest は最後に起動した run だけを指す。
            # 親ディレクトリ配下の run_*/slack を毎回列挙して、各 run の pending を拾う。
            spools = discover_spools(args.spool_root)
        processed = process_all_once(spools, token, channel_id, threads, args.dry_run)
        if args.once:
            print(f"processed {processed} records")
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
