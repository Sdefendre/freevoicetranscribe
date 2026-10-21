"""Resilient global hold-to-talk listener for macOS."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import Quartz

from .storage import normalize_record_key

FN_FLAG = 0x800000
# Device-dependent bits from IOKit so left and right modifiers stay distinct.
RIGHT_COMMAND_FLAG = 0x00000010
RIGHT_OPTION_FLAG = 0x00000040
RIGHT_CONTROL_FLAG = 0x00002000
SPACE_KEYCODE = 49
ESCAPE_KEYCODE = 53
F13_KEYCODE = 105


@dataclass(frozen=True, slots=True)
class _RecordKeySpec:
    flag: int | None = None
    keycode: int | None = None


# How each saved hotkey id is recognized in the event tap.
_RECORD_KEY_SPECS = {
    "fn": _RecordKeySpec(flag=FN_FLAG),
    "right-option": _RecordKeySpec(flag=RIGHT_OPTION_FLAG),
    "right-command": _RecordKeySpec(flag=RIGHT_COMMAND_FLAG),
    "right-control": _RecordKeySpec(flag=RIGHT_CONTROL_FLAG),
    "f13": _RecordKeySpec(keycode=F13_KEYCODE),
}


class FnListener:
    """Translate the event tap into serialized semantic callbacks."""

    def __init__(
        self,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_fn_space: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        *,
        record_key: str = "fn",
        startup_timeout: float = 2.0,
        stop_timeout: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.on_fn_space = on_fn_space
        self.on_cancel = on_cancel
        self.last_error: str | None = None
        self._logger = logger or logging.getLogger("freevoicetranscribe.hotkey")
        self._startup_timeout = startup_timeout
        self._stop_timeout = stop_timeout
        self._record_key_id = normalize_record_key(record_key)

        self._key_down = False
        self._recording_active = False
        self._events: queue.Queue[str | None] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._tap_thread: threading.Thread | None = None
        self._run_loop = None
        self._tap = None
        self._source = None
        self._ready = threading.Event()
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()

    @property
    def is_healthy(self) -> bool:
        thread = self._tap_thread
        return bool(thread and thread.is_alive() and self._running.is_set())

    @property
    def record_key(self) -> str:
        return self._record_key_id

    def set_record_key(self, key_id: str) -> None:
        key_id = normalize_record_key(key_id)
        if key_id == self._record_key_id:
            return
        self._record_key_id = key_id
        self._key_down = False

    def set_recording_active(self, active: bool) -> None:
        self._recording_active = bool(active)

    def _spec(self) -> _RecordKeySpec:
        return _RECORD_KEY_SPECS.get(self._record_key_id) or _RECORD_KEY_SPECS["fn"]

    def _dispatch(self, action: str) -> None:
        self._events.put(action)

    def _set_key_down(self, pressed: bool) -> None:
        if pressed and not self._key_down:
            self._key_down = True
            self._dispatch("press")
        elif not pressed and self._key_down:
            self._key_down = False
            self._dispatch("release")

    def _event_worker(self) -> None:
        while True:
            action = self._events.get()
            if action is None:
                return
            try:
                if action == "press":
                    self.on_press()
                elif action == "release":
                    self.on_release()
                elif action == "hands_free" and self.on_fn_space:
                    self.on_fn_space()
                elif action == "cancel" and self.on_cancel:
                    self.on_cancel()
            except Exception:
                self._logger.exception("Hotkey callback failed")

    def _callback(self, proxy, event_type, event, refcon):
        if event_type in {
            Quartz.kCGEventTapDisabledByTimeout,
            Quartz.kCGEventTapDisabledByUserInput,
        }:
            if self._tap is not None:
                Quartz.CGEventTapEnable(self._tap, True)
            self._logger.info("Re-enabled the global shortcut event tap")
            return event

        spec = self._spec()
        if event_type == Quartz.kCGEventFlagsChanged:
            if spec.flag is not None:
                self._set_key_down(bool(Quartz.CGEventGetFlags(event) & spec.flag))
            return event

        if event_type not in {Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp}:
            return event

        keycode = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventKeycode
        )
        if spec.keycode is not None and keycode == spec.keycode:
            # Swallow every event for the configured key, including auto-repeat,
            # so it cannot type into the focused app.
            is_repeat = False
            if event_type == Quartz.kCGEventKeyDown:
                is_repeat = bool(
                    Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventAutorepeat
                    )
                )
            if not is_repeat:
                self._set_key_down(event_type == Quartz.kCGEventKeyDown)
            return None
        if event_type != Quartz.kCGEventKeyDown:
            return event
        is_repeat = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGKeyboardEventAutorepeat
        )
        if is_repeat:
            return event
        if (
            self._recording_active
            and self._key_down
            and keycode == SPACE_KEYCODE
            and self.on_fn_space
        ):
            self._dispatch("hands_free")
            return None
        if self._recording_active and keycode == ESCAPE_KEYCODE and self.on_cancel:
            self._dispatch("cancel")
            return None
        return event

    def _create_tap(self):
        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged)
            | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventKeyUp)
        )
        return Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            self._callback,
            None,
        )

    def _run_tap(self) -> None:
        try:
            if self._tap is None:
                raise RuntimeError("The global shortcut event tap was not created")
            self._source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
            if self._source is None:
                raise RuntimeError("The global shortcut run-loop source failed")
            self._run_loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(
                self._run_loop, self._source, Quartz.kCFRunLoopDefaultMode
            )
            Quartz.CGEventTapEnable(self._tap, True)
            self._running.set()
            self._ready.set()
            Quartz.CFRunLoopRun()
        except Exception as exc:
            self.last_error = f"The global shortcut could not start: {exc}"
            self._logger.warning(
                "Global shortcut startup failed: %s", type(exc).__name__
            )
            self._ready.set()
        finally:
            self._running.clear()

    def start(self) -> bool:
        with self._lifecycle_lock:
            if self.is_healthy:
                return True
            if any(
                thread and thread.is_alive()
                for thread in (self._tap_thread, self._worker_thread)
            ):
                self.stop()
                if any(
                    thread and thread.is_alive()
                    for thread in (self._tap_thread, self._worker_thread)
                ):
                    self.last_error = "The previous global shortcut is still stopping."
                    return False
            self.last_error = None
            self._events = queue.Queue()
            self._ready = threading.Event()
            self._running = threading.Event()
            self._tap = self._create_tap()
            if self._tap is None:
                self.last_error = (
                    "Accessibility access is required for the hold-to-talk shortcut."
                )
                return False
            self._worker_thread = threading.Thread(
                target=self._event_worker,
                name="fvt-hotkey-events",
                daemon=True,
            )
            self._tap_thread = threading.Thread(
                target=self._run_tap,
                name="fvt-hotkey-tap",
                daemon=True,
            )
            self._worker_thread.start()
            self._tap_thread.start()

        deadline = time.monotonic() + self._startup_timeout
        ready = self._ready.wait(max(0.0, deadline - time.monotonic()))
        if not ready or not self.is_healthy:
            if not ready:
                self.last_error = "The global shortcut did not become ready in time."
            self.stop()
            return False
        return True

    def stop(self) -> bool:
        with self._lifecycle_lock:
            self._recording_active = False
            if self._run_loop is not None:
                Quartz.CFRunLoopStop(self._run_loop)
            self._events.put(None)
            if self._tap is not None:
                try:
                    Quartz.CFMachPortInvalidate(self._tap)
                except Exception:
                    pass

            deadline = time.monotonic() + self._stop_timeout
            for thread in (self._tap_thread, self._worker_thread):
                if thread and thread is not threading.current_thread():
                    thread.join(max(0.0, deadline - time.monotonic()))
            stopped = not any(
                thread and thread.is_alive()
                for thread in (self._tap_thread, self._worker_thread)
            )
            self._key_down = False
            self._running.clear()
            if stopped:
                self._run_loop = None
                self._source = None
                self._tap = None
                self._tap_thread = None
                self._worker_thread = None
            return stopped
