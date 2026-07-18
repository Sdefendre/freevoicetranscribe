import unittest
from contextlib import contextmanager

from fvt.contracts import ScreenFrame, TargetApp
from fvt.insertion import (
    ClipboardSnapshot,
    ClipboardStageError,
    InsertionError,
    MacTextInserter,
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

    def capture_focused_editable(self, _pid):
        return self.focused

    def focused_element_matches(self, _pid, expected):
        return self.focused is expected


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


if __name__ == "__main__":
    unittest.main()
