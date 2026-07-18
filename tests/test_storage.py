import json
import tempfile
import unittest
from pathlib import Path

from fvt.storage import AppPaths, LastTranscriptStore


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
            stale.write_bytes(b"private audio")
            unrelated = paths.temp / "keep.txt"
            unrelated.write_text("keep")
            self.assertEqual(paths.remove_stale_audio(), 1)
            self.assertFalse(stale.exists())
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


if __name__ == "__main__":
    unittest.main()
