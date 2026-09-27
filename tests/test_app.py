import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fvt.app import create_application, main
from fvt.storage import LastTranscriptStore


class AppLaunchTests(unittest.TestCase):
    def test_real_composition_restores_saved_transcript(self):
        original_directory = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with (
                    patch.dict(
                        os.environ,
                        {
                            "FVT_APP_SUPPORT": str(root / "support"),
                            "FVT_CACHE_DIR": str(root / "cache"),
                            "FVT_LOG_DIR": str(root / "logs"),
                        },
                    ),
                    patch("fvt.app.StatusBarApp"),
                    patch("fvt.app.AppHelper.callAfter"),
                ):
                    LastTranscriptStore.default().save("Saved dictation")
                    coordinator, _ = create_application()
                    try:
                        self.assertTrue(coordinator.snapshot.has_last_transcript)
                    finally:
                        coordinator.shutdown()
        finally:
            os.chdir(original_directory)

    def test_launch_smoke_reports_ready_before_entering_event_loop(self):
        coordinator = Mock()
        status_bar = Mock()
        with (
            patch.dict(os.environ, {"FVT_LAUNCH_SMOKE": "1"}),
            patch("fvt.app.create_application", return_value=(coordinator, status_bar)),
            patch("builtins.print") as output,
        ):
            main()

        coordinator.start.assert_called_once_with()
        output.assert_called_once_with("FVT_APP_READY", flush=True)
        status_bar.run.assert_called_once_with()
        coordinator.shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
