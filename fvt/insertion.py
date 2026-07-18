"""Fail-closed target validation and exact-once clipboard insertion."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import AppKit
import Quartz
from Foundation import NSData
from pynput import keyboard as pynput_keyboard

from .contracts import InsertResult, ScreenFrame, TargetApp

try:
    import ApplicationServices as AX
except ImportError:  # pragma: no cover - present in the packaged macOS runtime
    AX = None


class InsertionError(RuntimeError):
    pass


class ClipboardStageError(InsertionError):
    """Staging failed after mutating the pasteboard.

    ``expected_change_count`` lets the caller restore only when no other app
    has changed the pasteboard since our failed staging attempt.
    """

    def __init__(self, message: str, expected_change_count: int) -> None:
        super().__init__(message)
        self.expected_change_count = expected_change_count


@dataclass(frozen=True, slots=True)
class ClipboardSnapshot:
    items: tuple[tuple[tuple[str, bytes], ...], ...]


class MacClipboard:
    """Minimal typed wrapper around NSPasteboard for deterministic tests."""

    def __init__(self, pasteboard=None) -> None:
        self._pasteboard = pasteboard or AppKit.NSPasteboard.generalPasteboard()

    @property
    def change_count(self) -> int:
        return int(self._pasteboard.changeCount())

    def snapshot(self) -> ClipboardSnapshot:
        snapshots: list[tuple[tuple[str, bytes], ...]] = []
        for item in self._pasteboard.pasteboardItems() or []:
            values: list[tuple[str, bytes]] = []
            for pb_type in item.types() or []:
                data = item.dataForType_(pb_type)
                if data is not None:
                    values.append((str(pb_type), bytes(data)))
            if values:
                snapshots.append(tuple(values))
        return ClipboardSnapshot(tuple(snapshots))

    def stage_text(self, text: str) -> int:
        self._pasteboard.clearContents()
        if not self._pasteboard.setString_forType_(text, AppKit.NSPasteboardTypeString):
            raise ClipboardStageError(
                "Could not put the transcript on the clipboard", self.change_count
            )
        return self.change_count

    def restore(self, snapshot: ClipboardSnapshot) -> None:
        self._pasteboard.clearContents()
        if not snapshot.items:
            return
        pasteboard_items = []
        for stored_item in snapshot.items:
            item = AppKit.NSPasteboardItem.alloc().init()
            for pb_type, payload in stored_item:
                data = NSData.dataWithBytes_length_(payload, len(payload))
                if not item.setData_forType_(data, pb_type):
                    raise InsertionError("Could not restore the previous clipboard")
            pasteboard_items.append(item)
        if pasteboard_items and not self._pasteboard.writeObjects_(pasteboard_items):
            raise InsertionError("Could not restore the previous clipboard")

    def restore_if_unchanged(
        self, snapshot: ClipboardSnapshot, expected_change_count: int
    ) -> bool:
        """Restore only while our staged contents are still current.

        NSPasteboard exposes a monotonic change count but no compare-and-swap.
        Keeping the guard and restoration in one method is the narrowest race
        window macOS permits; a newer value observed before restoration is
        always preserved.
        """
        if self.change_count != expected_change_count:
            return False
        self.restore(snapshot)
        return True

    def copy_text(self, text: str) -> None:
        self.stage_text(text)


class MacAccessibility:
    """Capture and revalidate the exact focused editable AX element."""

    @staticmethod
    def _attribute(element, name):
        if AX is None:
            return None
        try:
            error, value = AX.AXUIElementCopyAttributeValue(element, name, None)
        except Exception:
            return None
        return value if error == AX.kAXErrorSuccess else None

    @staticmethod
    def _settable(element, name) -> bool:
        if AX is None:
            return False
        try:
            error, result = AX.AXUIElementIsAttributeSettable(element, name, None)
        except Exception:
            return False
        return error == AX.kAXErrorSuccess and bool(result)

    def capture_focused_editable(self, pid: int):
        if AX is None:
            return None
        app = AX.AXUIElementCreateApplication(pid)
        element = self._attribute(app, AX.kAXFocusedUIElementAttribute)
        if element is None:
            return None
        enabled = self._attribute(element, AX.kAXEnabledAttribute)
        if enabled is False:
            return None
        if not (
            self._settable(element, AX.kAXValueAttribute)
            or self._settable(element, AX.kAXSelectedTextAttribute)
        ):
            return None
        return element

    def focused_element_matches(self, pid: int, expected) -> bool:
        current = self.capture_focused_editable(pid)
        if current is None:
            return False
        try:
            return bool(Quartz.CFEqual(current, expected))
        except Exception:
            return current == expected


class MacWindowLocator:
    """Resolve the display containing the captured app's front window."""

    def screen_for_pid(self, pid: int) -> ScreenFrame | None:
        try:
            windows = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly
                | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
        except Exception:
            return None
        bounds = None
        for window in windows or []:
            if int(window.get(Quartz.kCGWindowOwnerPID, -1)) != pid:
                continue
            if int(window.get(Quartz.kCGWindowLayer, 0)) != 0:
                continue
            candidate = window.get(Quartz.kCGWindowBounds)
            if (
                candidate
                and candidate.get("Width", 0) > 1
                and candidate.get("Height", 0) > 1
            ):
                bounds = candidate
                break
        if bounds is None:
            return None

        screens = list(AppKit.NSScreen.screens() or [])
        primary = AppKit.NSScreen.mainScreen()
        if not screens or primary is None:
            return None
        primary_height = float(primary.frame().size.height)
        window_rect = (
            float(bounds["X"]),
            float(bounds["Y"]),
            float(bounds["Width"]),
            float(bounds["Height"]),
        )
        best_screen = None
        best_overlap = 0.0
        for screen in screens:
            frame = screen.frame()
            quartz_rect = (
                float(frame.origin.x),
                primary_height - float(frame.origin.y + frame.size.height),
                float(frame.size.width),
                float(frame.size.height),
            )
            overlap = self._intersection_area(window_rect, quartz_rect)
            if overlap > best_overlap:
                best_overlap = overlap
                best_screen = screen
        if best_screen is None:
            return None
        visible = best_screen.visibleFrame()
        return ScreenFrame(
            x=float(visible.origin.x),
            y=float(visible.origin.y),
            width=float(visible.size.width),
            height=float(visible.size.height),
        )

    @staticmethod
    def _intersection_area(a, b) -> float:
        left = max(a[0], b[0])
        top = max(a[1], b[1])
        right = min(a[0] + a[2], b[0] + b[2])
        bottom = min(a[1] + a[3], b[1] + b[3])
        return max(0.0, right - left) * max(0.0, bottom - top)


class MacTextInserter:
    """Insert once into the captured editable element or fail closed."""

    def __init__(
        self,
        *,
        clipboard: MacClipboard | None = None,
        keyboard_controller=None,
        workspace=None,
        running_app_lookup: Callable[[int], object | None] | None = None,
        accessibility=None,
        window_locator=None,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self._clipboard = clipboard or MacClipboard()
        self._keyboard = keyboard_controller or pynput_keyboard.Controller()
        self._workspace = workspace or AppKit.NSWorkspace.sharedWorkspace()
        self._running_app_lookup = running_app_lookup or (
            AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_
        )
        self._accessibility = accessibility or MacAccessibility()
        self._window_locator = window_locator or MacWindowLocator()
        self._sleep = sleep
        self._logger = logger or logging.getLogger("freevoicetranscribe.insertion")
        self._own_pid = os.getpid()
        self._captured_key: tuple[int, str] | None = None
        self._captured_element = None

    def capture_target(self) -> TargetApp | None:
        self._clear_capture()
        app = self._workspace.frontmostApplication()
        if app is None:
            return None
        pid = int(app.processIdentifier())
        if pid == self._own_pid:
            return None
        bundle_id = str(app.bundleIdentifier() or "")
        focused = self._accessibility.capture_focused_editable(pid)
        if focused is None:
            return None
        self._captured_key = (pid, bundle_id)
        self._captured_element = focused
        name = str(app.localizedName() or "Application")
        return TargetApp(
            pid=pid,
            name=name,
            bundle_id=bundle_id,
            screen=self._window_locator.screen_for_pid(pid),
        )

    def _clear_capture(self) -> None:
        self._captured_key = None
        self._captured_element = None

    def _activate_valid_target(self, target: TargetApp):
        if (
            self._captured_key != (target.pid, target.bundle_id)
            or self._captured_element is None
        ):
            raise InsertionError(
                "The original text field could not be verified. The transcript was not inserted."
            )
        app = self._running_app_lookup(target.pid)
        if app is None or bool(app.isTerminated()):
            raise InsertionError(
                f"{target.name} is no longer available. The transcript was not typed elsewhere."
            )
        current_bundle_id = str(app.bundleIdentifier() or "")
        if target.bundle_id and current_bundle_id != target.bundle_id:
            raise InsertionError(
                "The original application could not be verified. The transcript was not inserted."
            )
        activated = app.activateWithOptions_(
            AppKit.NSApplicationActivateIgnoringOtherApps
        )
        if activated is False:
            raise InsertionError(
                f"Could not return to {target.name}. The transcript was not inserted."
            )
        self._sleep(0.08)
        frontmost = self._workspace.frontmostApplication()
        if frontmost is None or int(frontmost.processIdentifier()) != target.pid:
            raise InsertionError(
                f"Could not verify focus in {target.name}. The transcript was not inserted."
            )
        if not self._accessibility.focused_element_matches(
            target.pid, self._captured_element
        ):
            raise InsertionError(
                "The original text field lost focus. The transcript was not inserted elsewhere."
            )
        return app

    def insert(self, text: str, target: TargetApp) -> InsertResult:
        if not text:
            raise InsertionError("There is no transcript to insert")
        try:
            self._activate_valid_target(target)
            try:
                snapshot = self._clipboard.snapshot()
            except Exception as exc:
                raise InsertionError(
                    "The current clipboard could not be preserved, so no text was inserted."
                ) from exc

            try:
                staged_change = self._clipboard.stage_text(text)
            except ClipboardStageError as exc:
                try:
                    self._clipboard.restore_if_unchanged(
                        snapshot, exc.expected_change_count
                    )
                except Exception:
                    self._logger.warning(
                        "Clipboard restoration failed after staging error"
                    )
                raise

            paste_attempted = False
            try:
                self._sleep(0.04)
                paste_attempted = True
                with self._keyboard.pressed(pynput_keyboard.Key.cmd):
                    self._keyboard.press("v")
                    self._keyboard.release("v")
                self._sleep(0.20)
            except Exception as exc:
                try:
                    self._clipboard.restore_if_unchanged(snapshot, staged_change)
                except Exception:
                    self._logger.warning(
                        "Clipboard restoration failed after paste error"
                    )
                phase = (
                    "after paste was attempted" if paste_attempted else "before paste"
                )
                raise InsertionError(f"Text insertion failed {phase}") from exc

            try:
                restored = self._clipboard.restore_if_unchanged(snapshot, staged_change)
            except Exception:
                # Paste was already sent. Never type a second copy as fallback.
                self._logger.warning(
                    "Clipboard restoration failed after successful paste"
                )
                return InsertResult(
                    method="paste",
                    clipboard_restored=False,
                    warning="Text was inserted, but the previous clipboard could not be restored.",
                )
            if not restored:
                return InsertResult(
                    method="paste",
                    clipboard_restored=False,
                    warning="The clipboard changed during insertion, so the newer contents were preserved.",
                )
            return InsertResult(method="paste", clipboard_restored=True)
        finally:
            self._clear_capture()

    def copy(self, text: str) -> None:
        if not text:
            raise InsertionError("There is no transcript to copy")
        self._clipboard.copy_text(text)
