import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fvt.storage import AppPaths, DeviceSettingsStore, LastTranscriptStore


class StorageTests(unittest.TestCase):
    def _paths(self, root):
        return AppPaths(
            root / "support", root / "cache", root / "logs", root / "temporary"
        )

    def test_next_launch_removes_even_recent_crash_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self._paths(Path(temp))
            paths.create()
            stale = paths.temp / "recording-crash.wav"
            convert = paths.temp / "convert-interview.wav"
            chunk = paths.temp / "chunk-abc.wav"
            stale.write_bytes(b"private audio")
            convert.write_bytes(b"converted")
            chunk.write_bytes(b"chunk")
            unrelated = paths.temp / "keep.txt"
            unrelated.write_text("keep")
            self.assertEqual(paths.remove_stale_audio(), 3)
            self.assertFalse(stale.exists())
            self.assertFalse(convert.exists())
            self.assertFalse(chunk.exists())
            self.assertTrue(unrelated.exists())

    def test_last_only_store_overwrites_and_reloads_without_history(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "support" / "last-transcript.json"
            store = LastTranscriptStore(path)
            store.save("first private transcript")
            store.save("second private transcript")

            self.assertEqual(
                LastTranscriptStore(path).load(), "second private transcript"
            )
            payload = json.loads(path.read_text())
            self.assertEqual(payload["policy"], "last-only")
            self.assertNotIn("first private transcript", path.read_text())
            self.assertNotIn("history", payload)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_atomic_saves_do_not_double_close_fd_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            transcript = LastTranscriptStore(root / "last-transcript.json")
            settings = DeviceSettingsStore(root / "device-settings.json")
            with (
                patch("fvt.storage.os.replace", side_effect=OSError("replace failed")),
                patch("fvt.storage.os.close", wraps=os.close) as close_spy,
            ):
                with self.assertRaises(OSError):
                    transcript.save("hello")
                with self.assertRaises(OSError):
                    settings.save({"dictation_mode": "dictate"})
            close_spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
