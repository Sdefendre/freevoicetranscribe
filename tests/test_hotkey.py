import threading
import unittest
from unittest.mock import patch

import Quartz

from fvt.hotkey import (
    F13_KEYCODE,
    FN_FLAG,
    RIGHT_OPTION_FLAG,
    SPACE_KEYCODE,
    FnListener,
)


class HotkeyTests(unittest.TestCase):
    def _space_event(self, listener, *, active):
        actions = []
        listener._dispatch = actions.append
        listener._key_down = True
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

    def test_fn_flag_change_dispatches_press_and_release(self):
        listener = FnListener(lambda: None, lambda: None)
        actions = []
        listener._dispatch = actions.append
        event = object()

        with patch.object(Quartz, "CGEventGetFlags", return_value=FN_FLAG):
            listener._callback(None, Quartz.kCGEventFlagsChanged, event, None)
        with patch.object(Quartz, "CGEventGetFlags", return_value=0):
            listener._callback(None, Quartz.kCGEventFlagsChanged, event, None)

        self.assertEqual(actions, ["press", "release"])

    def test_right_option_ignores_fn_and_tracks_its_own_flag(self):
        listener = FnListener(lambda: None, lambda: None, record_key="right-option")
        actions = []
        listener._dispatch = actions.append
        event = object()

        with patch.object(Quartz, "CGEventGetFlags", return_value=FN_FLAG):
            listener._callback(None, Quartz.kCGEventFlagsChanged, event, None)
        self.assertEqual(actions, [])

        with patch.object(Quartz, "CGEventGetFlags", return_value=RIGHT_OPTION_FLAG):
            listener._callback(None, Quartz.kCGEventFlagsChanged, event, None)
        self.assertEqual(actions, ["press"])

    def _f13_keydown(self, listener, *, repeat):
        def field(_event, requested):
            if requested == Quartz.kCGKeyboardEventAutorepeat:
                return 1 if repeat else 0
            return F13_KEYCODE

        event = object()
        with patch.object(Quartz, "CGEventGetIntegerValueField", side_effect=field):
            result = listener._callback(None, Quartz.kCGEventKeyDown, event, None)
        return event, result

    def test_f13_keydown_is_suppressed_and_dispatches_press(self):
        listener = FnListener(lambda: None, lambda: None, record_key="f13")
        actions = []
        listener._dispatch = actions.append
        _event, result = self._f13_keydown(listener, repeat=False)
        self.assertIsNone(result)
        self.assertEqual(actions, ["press"])

    def test_f13_autorepeat_is_swallowed_without_extra_press(self):
        listener = FnListener(lambda: None, lambda: None, record_key="f13")
        actions = []
        listener._dispatch = actions.append
        _event, first = self._f13_keydown(listener, repeat=False)
        _event, repeated = self._f13_keydown(listener, repeat=True)
        self.assertIsNone(first)
        self.assertIsNone(repeated)
        self.assertEqual(actions, ["press"])

    def test_set_record_key_rejects_unknown_ids(self):
        listener = FnListener(lambda: None, lambda: None)
        listener.set_record_key("banana")
        self.assertEqual(listener.record_key, "fn")
        listener.set_record_key("f13")
        self.assertEqual(listener.record_key, "f13")


if __name__ == "__main__":
    unittest.main()
