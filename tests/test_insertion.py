import unittest
from contextlib import contextmanager
from unittest.mock import patch

import AppKit
import Quartz

from fvt.contracts import ScreenFrame, SelectionMismatchError, TargetApp
from fvt.insertion import (
    ClipboardSnapshot,
    ClipboardStageError,
    InsertionError,
    MacTextInserter,
    MacWindowLocator,
)


class FakeApplication:
    def __init__(self, pid=123, bundle_id="test.app", *, terminated=False):
        self.pid = pid
        self.bundle_id = bundle_id
        self.terminated = terminated

    def isTerminated(self):
        return self.terminated

    def bundleIdentifier(self):
        return self.bundle_id

    def processIdentifier(self):
        return self.pid

    def activateWithOptions_(self, _options):
        return True

    def localizedName(self):
        return "Test"


class FakeWorkspace:
    def __init__(self, app):
        self.app = app

    def frontmostApplication(self):
        return self.app


class FakeClipboard:
    def __init__(self, *, fail_stage=False):
        self.contents = "original"
        self._change_count = 1
        self.restored = False
        self.fail_stage = fail_stage

    @property
    def change_count(self):
        return self._change_count

    def snapshot(self):
        return ClipboardSnapshot(((("public.utf8-plain-text", b"original"),),))

    def stage_text(self, text):
        self.contents = "" if self.fail_stage else text
        self._change_count += 1
        if self.fail_stage:
            raise ClipboardStageError("staging failed", self._change_count)
        return self._change_count

    def restore(self, _snapshot):
        self.contents = "original"
        self._change_count += 1
        self.restored = True

    def restore_if_unchanged(self, snapshot, expected_change_count):
        if self._change_count != expected_change_count:
            return False
        self.restore(snapshot)
        return True

    def copy_text(self, text):
        self.contents = text
        self._change_count += 1


class FakeKeyboard:
    def __init__(self, on_paste=None):
        self.on_paste = on_paste

    @contextmanager
    def pressed(self, _key):
        yield

    def press(self, key):
        if key == "v" and self.on_paste:
            self.on_paste()

    def release(self, _key):
        pass


class FakeAccessibility:
    def __init__(self):
        self.focused = object()
        self.selected = "already highlighted"

    def capture_focused_editable(self, _pid):
        return self.focused

    def focused_element_matches(self, _pid, expected):
        return self.focused is expected

    def selected_text(self, element):
        return self.selected if element is self.focused else ""


class FakeWindowLocator:
    screen = ScreenFrame(x=100, y=200, width=1200, height=800)

    def screen_for_pid(self, _pid):
        return self.screen


class InsertionTests(unittest.TestCase):
    def _service(self, app, clipboard, keyboard):
        workspace = FakeWorkspace(app)
        accessibility = FakeAccessibility()
        service = MacTextInserter(
            clipboard=clipboard,
            keyboard_controller=keyboard,
            workspace=workspace,
            running_app_lookup=lambda _pid: app,
            accessibility=accessibility,
            window_locator=FakeWindowLocator(),
            sleep=lambda _seconds: None,
        )
        return service, accessibility

    def _capture(self, service):
        target = service.capture_target()
        self.assertIsNotNone(target)
        return target

    def test_paste_restores_unchanged_clipboard_exactly_once(self):
        app = FakeApplication()
        clipboard = FakeClipboard()
        service, _accessibility = self._service(app, clipboard, FakeKeyboard())
        target = self._capture(service)
        result = service.insert("hello", target)
        self.assertTrue(result.clipboard_restored)
        self.assertTrue(clipboard.restored)
        self.assertEqual(clipboard.contents, "original")

    def test_newer_clipboard_change_is_never_overwritten(self):
        app = FakeApplication()
        clipboard = FakeClipboard()

        def external_change():
            clipboard.contents = "newer"
            clipboard._change_count += 1

        service, _accessibility = self._service(
            app, clipboard, FakeKeyboard(external_change)
        )
        target = self._capture(service)
        result = service.insert("hello", target)
        self.assertFalse(result.clipboard_restored)
        self.assertFalse(clipboard.restored)
        self.assertEqual(clipboard.contents, "newer")

    def test_invalid_target_fails_before_touching_clipboard(self):
        app = FakeApplication(terminated=True)
        clipboard = FakeClipboard()
        service, _accessibility = self._service(app, clipboard, FakeKeyboard())
        target = TargetApp(pid=123, name="Test", bundle_id="test.app")
        with self.assertRaises(InsertionError):
            service.insert("hello", target)
        self.assertEqual(clipboard.contents, "original")
        self.assertFalse(clipboard.restored)

    def test_staging_failure_restores_original_clipboard(self):
        app = FakeApplication()
        clipboard = FakeClipboard(fail_stage=True)
        service, _accessibility = self._service(app, clipboard, FakeKeyboard())
        target = self._capture(service)
        with self.assertRaises(ClipboardStageError):
            service.insert("hello", target)
        self.assertTrue(clipboard.restored)
        self.assertEqual(clipboard.contents, "original")

    def test_focus_change_fails_closed_before_clipboard_mutation(self):
        app = FakeApplication()
        clipboard = FakeClipboard()
        service, accessibility = self._service(app, clipboard, FakeKeyboard())
        target = self._capture(service)
        accessibility.focused = object()
        with self.assertRaisesRegex(InsertionError, "lost focus"):
            service.insert("hello", target)
        self.assertEqual(clipboard.contents, "original")
        self.assertFalse(clipboard.restored)

    def test_capture_uses_target_window_display(self):
        service, _accessibility = self._service(
            FakeApplication(), FakeClipboard(), FakeKeyboard()
        )
        target = self._capture(service)
        self.assertEqual(target.screen, FakeWindowLocator.screen)

    def test_read_selected_text_uses_the_captured_element(self):
        service, _accessibility = self._service(
            FakeApplication(), FakeClipboard(), FakeKeyboard()
        )
        target = self._capture(service)
        self.assertEqual(service.read_selected_text(target), "already highlighted")

    def test_read_selected_text_is_empty_without_a_matching_capture(self):
        service, _accessibility = self._service(
            FakeApplication(), FakeClipboard(), FakeKeyboard()
        )
        target = TargetApp(pid=123, name="Test", bundle_id="test.app")
        self.assertEqual(service.read_selected_text(target), "")

    def test_insert_aborts_before_paste_when_selection_changes(self):
        app = FakeApplication()
        clipboard = FakeClipboard()
        service, accessibility = self._service(app, clipboard, FakeKeyboard())
        target = self._capture(service)
        accessibility.selected = "changed later"
        with self.assertRaises(SelectionMismatchError):
            service.insert("rewrite", target, expected_selection="already highlighted")
        self.assertEqual(clipboard.contents, "original")
        self.assertFalse(clipboard.restored)

    def test_window_locator_uses_primary_screen_height_not_main_screen(self):
        class FakeScreen:
            def __init__(self, frame, visible):
                self._frame = frame
                self._visible = visible

            def frame(self):
                return self._frame

            def visibleFrame(self):
                return self._visible

        # Primary screen (screens[0]) is 1440x900 at Cocoa (0, 0).
        # Secondary tall monitor is 2560x1440 placed to the right, bottom-aligned
        # at Cocoa y=0 (Quartz y = 900 - 1440 = -540).
        primary = FakeScreen(
            AppKit.NSMakeRect(0, 0, 1440, 900),
            AppKit.NSMakeRect(0, 0, 1440, 875),
        )
        secondary = FakeScreen(
            AppKit.NSMakeRect(1440, 0, 2560, 1440),
            AppKit.NSMakeRect(1440, 0, 2560, 1415),
        )
        # Window sits near the top of the secondary monitor (Quartz y = -400).
        windows = [
            {
                Quartz.kCGWindowOwnerPID: 4242,
                Quartz.kCGWindowLayer: 0,
                Quartz.kCGWindowBounds: {
                    "X": 1600,
                    "Y": -400,
                    "Width": 800,
                    "Height": 300,
                },
            }
        ]

        class FakeNSScreen:
            @staticmethod
            def screens():
                return [primary, secondary]

            @staticmethod
            def mainScreen():
                return secondary

        fake_appkit = type("FakeAppKit", (), {"NSScreen": FakeNSScreen})
        with (
            patch.object(Quartz, "CGWindowListCopyWindowInfo", return_value=windows),
            patch("fvt.insertion.AppKit", fake_appkit),
        ):
            frame = MacWindowLocator().screen_for_pid(4242)
        self.assertEqual(
            frame,
            ScreenFrame(x=1440.0, y=0.0, width=2560.0, height=1415.0),
        )


if __name__ == "__main__":
    unittest.main()
