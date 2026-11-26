import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fvt.app import create_application, main
from fvt.storage import AppPaths


class AppLaunchTests(unittest.TestCase):
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

    def test_create_application_uses_provided_paths_for_last_transcript_store(self):
        logger = logging.getLogger("freevoicetranscribe")
        existing_handlers = list(logger.handlers)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, {}, clear=False),
        ):
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "temp"
            )
            cwd = os.getcwd()
            try:
                with patch("fvt.app.StatusBarApp") as status_bar_cls:
                    status_bar_cls.return_value = Mock()
                    coordinator, _status_bar = create_application(paths=paths)
                self.assertEqual(
                    coordinator._transcript_store.path,
                    paths.support / "last-transcript.json",
                )
                coordinator.shutdown()
            finally:
                os.chdir(cwd)
                for handler in list(logger.handlers):
                    if handler not in existing_handlers:
                        logger.removeHandler(handler)
                        handler.close()


if __name__ == "__main__":
    unittest.main()
