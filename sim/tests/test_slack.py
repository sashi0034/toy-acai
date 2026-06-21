import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sim import _slack as slack
from sim import slack_uploader


class SlackPosterTest(unittest.TestCase):
    def test_direct_poster_attaches_hyperparameters_then_posts_gif_to_thread(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hyperparameters = root / "hyperparameters.py"
            hyperparameters.write_text("NUM_UPDATES = 1\n", encoding="utf-8")
            gif_path = root / "update.gif"
            gif_path.write_bytes(b"GIF89a")
            fake_thread = mock.MagicMock()
            fake_thread.token = "token"
            fake_thread.channel_id = "channel"
            fake_thread.ensure.return_value = "123.456"

            with mock.patch.object(slack, "SlackThread", return_value=fake_thread), mock.patch.object(
                slack, "upload_file", return_value="FGIF"
            ) as upload, mock.patch.dict(
                os.environ, {"SLACK_BOT_TOKEN": "token", "SLACK_CHANNEL_ID": "channel"}
            ):
                poster = slack.DirectPoster("token", "channel", hyperparameters)
                poster.start()
                poster.post_file(gif_path, "update", "ignored")

            fake_thread.post_root.assert_called_once_with(
                "toy-acai sim training started", hyperparameters
            )
            upload.assert_called_once_with("token", "channel", gif_path, "update", "123.456")

    def test_off_backend_creates_no_spool(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "output"
            with mock.patch.dict(os.environ, {"SLACK_POST_BACKEND": "off"}):
                poster = slack.create_poster(output, root)
                poster.start()
                poster.post_file(root / "missing.gif", "ignored", "ignored")

            self.assertFalse((output / "slack").exists())

    def test_external_poster_writes_root_then_gif_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hyperparameters = root / "hyperparameters.py"
            hyperparameters.write_text("NUM_UPDATES = 1\n", encoding="utf-8")
            gif_path = root / "update.gif"
            gif_path.write_bytes(b"GIF89a")

            poster = slack.ExternalPoster(root / "run" / "slack", hyperparameters)
            poster.start()
            poster.post_file(gif_path, "update", "update_000000_0000")

            pending = root / "run" / "slack" / "pending"
            records = sorted(pending.glob("*.json"))
            self.assertEqual([path.name for path in records], [
                "000000_thread_root.json",
                "update_000000_0000.json",
            ])
            root_record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertEqual(root_record["type"], "thread_root")
            self.assertEqual(root_record["attachment_path"], str(hyperparameters.resolve()))

    def test_uploader_sends_root_and_attachment_to_its_thread(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            spool = root / "run" / "slack"
            hyperparameters = root / "hyperparameters.py"
            hyperparameters.write_text("NUM_UPDATES = 1\n", encoding="utf-8")
            gif_path = root / "update.gif"
            gif_path.write_bytes(b"GIF89a")
            slack.write_spool_record(
                spool,
                "000000_thread_root",
                {"type": "thread_root", "attachment_path": str(hyperparameters), "comment": "start"},
            )
            slack.write_spool_record(
                spool,
                "update_000000_0000",
                {"file_path": str(gif_path), "comment": "update"},
            )
            uploads = []

            def fake_upload(_token, _channel, path, initial_comment="", thread_ts=None):
                uploads.append((path.name, initial_comment, thread_ts))
                return "FROOT"

            thread = slack.SlackThread("token", "channel", spool / "thread_ts")
            with mock.patch.object(slack_uploader, "upload_file", side_effect=fake_upload), mock.patch.object(
                slack_uploader, "file_share_ts", return_value="123.456"
            ):
                processed = slack_uploader.process_once(spool, thread)

            self.assertEqual(processed, 2)
            self.assertEqual(
                uploads,
                [("hyperparameters.py", "start", None), ("update.gif", "update", "123.456")],
            )
            self.assertEqual((spool / "thread_ts").read_text(encoding="utf-8").strip(), "123.456")
            self.assertEqual(len(list((spool / "sent").glob("*.json"))), 2)

    def test_discover_spools_uses_current_timestamped_output_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "20260621_120000"
            second = root / "20260621_120001"
            (first / "slack").mkdir(parents=True)
            (second / "slack").mkdir(parents=True)
            (root / "not-a-run").mkdir()

            self.assertEqual(
                slack_uploader.discover_spools(root),
                [first / "slack", second / "slack"],
            )


if __name__ == "__main__":
    unittest.main()
