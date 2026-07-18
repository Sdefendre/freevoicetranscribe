import os
import unittest
from unittest.mock import Mock, patch

from fvt.app import main


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


if __name__ == "__main__":
    unittest.main()
