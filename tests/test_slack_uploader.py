import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

import slack_uploader  # noqa: E402


class SlackUploaderTest(unittest.TestCase):
    def test_thread_root_uploads_markdown_file_and_saves_share_ts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spool = root / "spool"
            overview = root / "rl_model_overview.md"
            overview.write_text("# overview\n", encoding="utf-8")
            record_path = root / "thread_root.json"
            record_path.write_text(
                json.dumps(
                    {
                        "type": "thread_root",
                        "comment": "training started",
                        "attachment_path": str(overview),
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_slack_request(method, _token, payload):
                calls.append((method, payload))
                if method == "files.getUploadURLExternal":
                    return {"upload_url": "https://upload.example", "file_id": "FROOT"}
                if method == "files.completeUploadExternal":
                    return {"ok": True, "files": [{"id": "FROOT"}]}
                if method == "files.info":
                    return {
                        "file": {
                            "shares": {
                                "public": {
                                    "C123": [
                                        {
                                            "ts": "1710000000.000100",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                raise AssertionError(f"unexpected Slack method: {method}")

            uploaded = []
            original_slack_request = slack_uploader.slack_request
            original_upload_bytes = slack_uploader.upload_bytes
            slack_uploader.slack_request = fake_slack_request
            slack_uploader.upload_bytes = lambda _url, path: uploaded.append(path)
            try:
                thread = slack_uploader.SlackThread(spool, "xoxb-token", "C123", dry_run=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    slack_uploader.upload_record(record_path, "xoxb-token", "C123", thread, dry_run=False)
            finally:
                slack_uploader.slack_request = original_slack_request
                slack_uploader.upload_bytes = original_upload_bytes

            self.assertEqual(thread.thread_ts, "1710000000.000100")
            self.assertEqual((spool / "thread_ts").read_text(encoding="utf-8").strip(), "1710000000.000100")
            updated_record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(updated_record["uploaded_file_id"], "FROOT")
            self.assertEqual(uploaded, [overview])
            self.assertIn(("files.info", {"file": "FROOT"}), calls)

    def test_thread_root_reuses_recorded_file_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spool = root / "spool"
            overview = root / "rl_model_overview.md"
            overview.write_text("# overview\n", encoding="utf-8")
            record_path = root / "thread_root.json"
            record_path.write_text(
                json.dumps(
                    {
                        "type": "thread_root",
                        "comment": "training started",
                        "attachment_path": str(overview),
                        "uploaded_file_id": "FROOT",
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_slack_request(method, _token, payload):
                calls.append((method, payload))
                if method == "files.info":
                    return {
                        "file": {
                            "shares": {
                                "public": {
                                    "C123": [
                                        {
                                            "ts": "1710000000.000100",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                raise AssertionError(f"unexpected Slack method: {method}")

            original_slack_request = slack_uploader.slack_request
            original_upload_file = slack_uploader.upload_file
            slack_uploader.slack_request = fake_slack_request
            slack_uploader.upload_file = lambda *_args, **_kwargs: self.fail("should not upload again")
            try:
                thread = slack_uploader.SlackThread(spool, "xoxb-token", "C123", dry_run=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    slack_uploader.upload_record(record_path, "xoxb-token", "C123", thread, dry_run=False)
            finally:
                slack_uploader.slack_request = original_slack_request
                slack_uploader.upload_file = original_upload_file

            self.assertEqual(thread.thread_ts, "1710000000.000100")
            self.assertEqual(calls, [("files.info", {"file": "FROOT"})])

    def test_gif_upload_uses_existing_thread_ts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spool = root / "spool"
            spool.mkdir()
            (spool / "thread_ts").write_text("1710000000.000100\n", encoding="utf-8")
            gif_path = root / "episode.gif"
            gif_path.write_bytes(b"gif")
            record_path = root / "episode.json"
            record_path.write_text(
                json.dumps({"gif_path": str(gif_path), "comment": "episode"}),
                encoding="utf-8",
            )
            uploads = []

            original_upload_file = slack_uploader.upload_file
            slack_uploader.upload_file = (
                lambda token, channel_id, file_path, initial_comment="", thread_ts=None: uploads.append(
                    (token, channel_id, file_path, initial_comment, thread_ts)
                )
                or "FGIF"
            )
            try:
                thread = slack_uploader.SlackThread(spool, "xoxb-token", "C123", dry_run=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    slack_uploader.upload_record(record_path, "xoxb-token", "C123", thread, dry_run=False)
            finally:
                slack_uploader.upload_file = original_upload_file

            self.assertEqual(
                uploads,
                [("xoxb-token", "C123", gif_path, "episode", "1710000000.000100")],
            )

    def test_file_path_upload_uses_existing_thread_ts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spool = root / "spool"
            spool.mkdir()
            (spool / "thread_ts").write_text("1710000000.000100\n", encoding="utf-8")
            chart_path = root / "reward_chart.png"
            chart_path.write_bytes(b"png")
            record_path = root / "reward_chart.json"
            record_path.write_text(
                json.dumps({"file_path": str(chart_path), "comment": "reward trend"}),
                encoding="utf-8",
            )
            uploads = []

            original_upload_file = slack_uploader.upload_file
            slack_uploader.upload_file = (
                lambda token, channel_id, file_path, initial_comment="", thread_ts=None: uploads.append(
                    (token, channel_id, file_path, initial_comment, thread_ts)
                )
                or "FPNG"
            )
            try:
                thread = slack_uploader.SlackThread(spool, "xoxb-token", "C123", dry_run=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    slack_uploader.upload_record(record_path, "xoxb-token", "C123", thread, dry_run=False)
            finally:
                slack_uploader.upload_file = original_upload_file

            self.assertEqual(
                uploads,
                [("xoxb-token", "C123", chart_path, "reward trend", "1710000000.000100")],
            )

    def test_process_all_once_uploads_records_from_multiple_run_spools(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spools = []
            for index, thread_ts in enumerate(["1710000000.000100", "1710000000.000200"], start=1):
                spool = root / f"run_20260616_00000{index}" / "slack"
                pending = spool / "pending"
                pending.mkdir(parents=True)
                (spool / "thread_ts").write_text(f"{thread_ts}\n", encoding="utf-8")
                gif_path = root / f"episode_{index}.gif"
                gif_path.write_bytes(b"gif")
                (pending / f"episode_{index}.json").write_text(
                    json.dumps({"gif_path": str(gif_path), "comment": f"episode {index}"}),
                    encoding="utf-8",
                )
                spools.append(spool)

            uploads = []
            original_upload_file = slack_uploader.upload_file
            slack_uploader.upload_file = (
                lambda token, channel_id, file_path, initial_comment="", thread_ts=None: uploads.append(
                    (token, channel_id, file_path.name, initial_comment, thread_ts)
                )
                or "FGIF"
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    processed = slack_uploader.process_all_once(
                        spools,
                        "xoxb-token",
                        "C123",
                        {},
                        dry_run=False,
                    )
            finally:
                slack_uploader.upload_file = original_upload_file

            self.assertEqual(processed, 2)
            self.assertEqual(
                uploads,
                [
                    ("xoxb-token", "C123", "episode_1.gif", "episode 1", "1710000000.000100"),
                    ("xoxb-token", "C123", "episode_2.gif", "episode 2", "1710000000.000200"),
                ],
            )
            self.assertTrue((spools[0] / "sent" / "episode_1.json").exists())
            self.assertTrue((spools[1] / "sent" / "episode_2.json").exists())

    def test_discover_spools_lists_run_dirs_without_latest_symlink(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "run_20260616_000001"
            second = root / "run_20260616_000002"
            (first / "slack").mkdir(parents=True)
            (second / "slack").mkdir(parents=True)
            (root / "latest").symlink_to(second.name)

            self.assertEqual(
                slack_uploader.discover_spools(root),
                [first / "slack", second / "slack"],
            )

    def test_file_share_ts_retries_until_slack_share_is_visible(self):
        responses = [
            {"file": {"shares": {}}},
            {
                "file": {
                    "shares": {
                        "private": {
                            "C123": [
                                {
                                    "ts": "1710000000.000200",
                                }
                            ]
                        }
                    }
                }
            },
        ]
        sleeps = []

        original_slack_request = slack_uploader.slack_request
        original_sleep = slack_uploader.time.sleep
        slack_uploader.slack_request = lambda _method, _token, _payload: responses.pop(0)
        slack_uploader.time.sleep = lambda seconds: sleeps.append(seconds)
        try:
            ts = slack_uploader.file_share_ts(
                "xoxb-token",
                "FROOT",
                "C123",
                attempts=2,
                delay_seconds=0.25,
            )
        finally:
            slack_uploader.slack_request = original_slack_request
            slack_uploader.time.sleep = original_sleep

        self.assertEqual(ts, "1710000000.000200")
        self.assertEqual(sleeps, [0.25])


if __name__ == "__main__":
    unittest.main()
