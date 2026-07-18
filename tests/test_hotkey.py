import threading
import unittest
from unittest.mock import patch

import Quartz

from fvt.hotkey import SPACE_KEYCODE, FnListener


class HotkeyTests(unittest.TestCase):
    def _space_event(self, listener, *, active):
        actions = []
        listener._dispatch = actions.append
        listener._fn_down = True
        listener.set_recording_active(active)

        def field(_event, requested):
            if requested == Quartz.kCGKeyboardEventAutorepeat:
                return 0
            return SPACE_KEYCODE

        event = object()
        with patch.object(Quartz, "CGEventGetIntegerValueField", side_effect=field):
            result = listener._callback(None, Quartz.kCGEventKeyDown, event, None)
        return event, result, actions

    def test_fn_space_is_not_suppressed_while_idle(self):
        listener = FnListener(lambda: None, lambda: None, lambda: None)
        event, result, actions = self._space_event(listener, active=False)
        self.assertIs(result, event)
        self.assertEqual(actions, [])

    def test_fn_space_is_suppressed_only_during_active_recording(self):
        listener = FnListener(lambda: None, lambda: None, lambda: None)
        _event, result, actions = self._space_event(listener, active=True)
        self.assertIsNone(result)
        self.assertEqual(actions, ["hands_free"])

    def test_start_waits_for_run_loop_health_and_stop_is_bounded(self):
        listener = FnListener(
            lambda: None,
            lambda: None,
            startup_timeout=0.2,
            stop_timeout=0.2,
        )
        run_loop_stopped = threading.Event()
        with (
            patch.object(listener, "_create_tap", return_value=object()),
            patch.object(
                Quartz, "CFMachPortCreateRunLoopSource", return_value=object()
            ),
            patch.object(Quartz, "CFRunLoopGetCurrent", return_value=object()),
            patch.object(Quartz, "CFRunLoopAddSource"),
            patch.object(Quartz, "CGEventTapEnable"),
            patch.object(
                Quartz,
                "CFRunLoopRun",
                side_effect=lambda: run_loop_stopped.wait(1.0),
            ),
            patch.object(
                Quartz,
                "CFRunLoopStop",
                side_effect=lambda _loop: run_loop_stopped.set(),
            ),
            patch.object(Quartz, "CFMachPortInvalidate"),
        ):
            self.assertTrue(listener.start())
            self.assertTrue(listener.is_healthy)
            self.assertTrue(listener.stop())
            self.assertFalse(listener.is_healthy)

    def test_startup_failure_is_reported_before_start_returns(self):
        listener = FnListener(
            lambda: None,
            lambda: None,
            startup_timeout=0.2,
            stop_timeout=0.2,
        )
        with (
            patch.object(listener, "_create_tap", return_value=object()),
            patch.object(
                Quartz,
                "CFMachPortCreateRunLoopSource",
                side_effect=RuntimeError("source failed"),
            ),
            patch.object(Quartz, "CFMachPortInvalidate"),
        ):
            self.assertFalse(listener.start())
        self.assertFalse(listener.is_healthy)
        self.assertIn("could not start", listener.last_error.lower())


if __name__ == "__main__":
    unittest.main()
