"""Slack posting helpers for training runs.

The ``external`` backend writes small JSON records to a run-local spool.  The
uploader can then run on a login node that has network access, while the
training process on a compute node never opens a network connection.
"""

from datetime import datetime
import json
import mimetypes
import os
from pathlib import Path
import socket
import time
from typing import Protocol
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo


SLACK_API = "https://slack.com/api"
BACKENDS = {"off", "direct", "external"}


def training_started_message() -> str:
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S %Z")
    return f"toy-acai started at {timestamp} on {socket.gethostname()}"


def load_dotenv(path: Path) -> None:
    """Load a small, dependency-free subset of dotenv syntax.

    Existing environment variables deliberately win so a scheduler can supply
    credentials without changing the checked-out repository.
    """
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
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        os.environ[key] = value


def slack_backend(repository_root: Path) -> str:
    load_dotenv(repository_root / ".env")
    backend = os.environ.get("SLACK_POST_BACKEND", "off").lower()
    if backend not in BACKENDS:
        raise ValueError(
            "SLACK_POST_BACKEND must be one of " + ", ".join(sorted(BACKENDS))
        )
    return backend


def slack_request(method: str, token: str, payload: dict[str, str]) -> dict:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{SLACK_API}/{method}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {result}")
    return result


def post_message(token: str, channel_id: str, text: str, thread_ts: str | None = None) -> str:
    payload = {"channel": channel_id, "text": text}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    return str(slack_request("chat.postMessage", token, payload)["ts"])


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
    thread_ts: str | None = None,
) -> str:
    upload = slack_request(
        "files.getUploadURLExternal",
        token,
        {"filename": file_path.name, "length": str(file_path.stat().st_size)},
    )
    upload_bytes(str(upload["upload_url"]), file_path)
    payload = {
        "files": json.dumps([{"id": upload["file_id"], "title": file_path.name}]),
        "channel_id": channel_id,
        "initial_comment": initial_comment,
    }
    if thread_ts:
        payload["thread_ts"] = thread_ts
    slack_request("files.completeUploadExternal", token, payload)
    return str(upload["file_id"])


def _entry_share_ts(entry: object) -> str | None:
    if not isinstance(entry, dict):
        return None
    thread_ts = entry.get("ts") or entry.get("thread_ts")
    return str(thread_ts) if thread_ts else None


def _walk_entries(value: object):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_entries(child)


def _share_ts(shares: object, channel_id: str) -> str | None:
    if not isinstance(shares, dict):
        return None
    for visibility in ("public", "private"):
        by_channel = shares.get(visibility, {})
        if not isinstance(by_channel, dict):
            continue
        for entry in by_channel.get(channel_id, []):
            thread_ts = _entry_share_ts(entry)
            if thread_ts:
                return thread_ts
    for entry in _walk_entries(shares):
        if entry.get("channel") not in (channel_id,) and entry.get("channel_id") not in (
            channel_id,
        ):
            continue
        thread_ts = _entry_share_ts(entry)
        if thread_ts:
            return thread_ts
    return None


def file_share_ts(
    token: str,
    file_id: str,
    channel_id: str,
    attempts: int = 8,
    delay_seconds: float = 2.0,
) -> str:
    for attempt in range(max(1, attempts)):
        result = slack_request("files.info", token, {"file": file_id})
        thread_ts = _share_ts(result.get("file", {}).get("shares", {}), channel_id)
        if thread_ts is not None:
            return thread_ts
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(f"could not find Slack share ts for file {file_id} in {channel_id}")


class SlackThread:
    def __init__(
        self,
        token: str,
        channel_id: str,
        state_path: Path | None = None,
        dry_run: bool = False,
    ):
        self.token = token
        self.channel_id = channel_id
        self.state_path = state_path
        self.dry_run = dry_run
        self.thread_ts = self._read_thread_ts()

    def _read_thread_ts(self) -> str | None:
        if self.state_path is None or not self.state_path.exists():
            return None
        thread_ts = self.state_path.read_text(encoding="utf-8").strip()
        return thread_ts or None

    def _write_thread_ts(self, thread_ts: str) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(f"{thread_ts}\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def post_root(self, comment: str, attachment_path: Path | None = None) -> str:
        if self.dry_run:
            self.thread_ts = self.thread_ts or "dry-run-thread"
            return self.thread_ts
        if attachment_path is None:
            self.thread_ts = post_message(self.token, self.channel_id, comment)
        else:
            if not attachment_path.exists():
                raise FileNotFoundError(f"Slack attachment does not exist: {attachment_path}")
            file_id = upload_file(self.token, self.channel_id, attachment_path, comment)
            self.thread_ts = file_share_ts(self.token, file_id, self.channel_id)
        self._write_thread_ts(self.thread_ts)
        return self.thread_ts

    def ensure(self) -> str:
        return self.thread_ts or self.post_root("toy-acai sim training updates")


def write_spool_record(spool: Path, name: str, payload: dict) -> None:
    pending = spool / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    destination = pending / f"{name}.json"
    temporary = pending / f".{name}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)


class Poster(Protocol):
    def start(self) -> None: ...

    def post_file(self, file_path: Path, comment: str, record_name: str) -> None: ...


class NullPoster:
    def start(self) -> None:
        pass

    def post_file(self, file_path: Path, comment: str, record_name: str) -> None:
        pass


class DirectPoster:
    def __init__(self, token: str | None, channel_id: str | None, hyperparameters_path: Path):
        self.token = token
        self.channel_id = channel_id
        self.hyperparameters_path = hyperparameters_path
        self.thread: SlackThread | None = None
        self.disabled = False

    def _warn(self, message: str) -> None:
        print(f"Slack direct posting disabled: {message}")

    def start(self) -> None:
        if not self.token or not self.channel_id:
            self.disabled = True
            self._warn("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID are required")
            return
        try:
            self.thread = SlackThread(self.token, self.channel_id)
            self.thread.post_root(training_started_message(), self.hyperparameters_path)
        except Exception as error:
            self.disabled = True
            self._warn(str(error))

    def post_file(self, file_path: Path, comment: str, record_name: str) -> None:
        del record_name
        if self.disabled or self.thread is None:
            return
        try:
            upload_file(
                self.thread.token,
                self.thread.channel_id,
                file_path,
                comment,
                self.thread.ensure(),
            )
        except Exception as error:
            print(f"Slack direct posting failed for {file_path}: {error}")


class ExternalPoster:
    def __init__(self, spool: Path, hyperparameters_path: Path):
        self.spool = spool
        self.hyperparameters_path = hyperparameters_path

    def start(self) -> None:
        write_spool_record(
            self.spool,
            "000000_thread_root",
            {
                "type": "thread_root",
                "attachment_path": str(self.hyperparameters_path.resolve()),
                "comment": training_started_message(),
            },
        )

    def post_file(self, file_path: Path, comment: str, record_name: str) -> None:
        write_spool_record(
            self.spool,
            record_name,
            {"file_path": str(file_path.resolve()), "comment": comment},
        )


def create_poster(output_directory: Path, repository_root: Path) -> Poster:
    backend = slack_backend(repository_root)
    if backend == "off":
        return NullPoster()
    hyperparameters_path = repository_root / "sim" / "hyperparameters.py"
    if backend == "external":
        return ExternalPoster(output_directory / "slack", hyperparameters_path)
    return DirectPoster(
        os.environ.get("SLACK_BOT_TOKEN"),
        os.environ.get("SLACK_CHANNEL_ID"),
        hyperparameters_path,
    )
