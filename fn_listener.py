"""
Fn key hold-to-talk listener using Quartz CGEventTap.

pynput cannot detect the Fn key on macOS, so we use a low-level
CGEventTap to monitor flagsChanged events and check the
kCGEventFlagMaskSecondaryFn bit (0x800000).

Requires Accessibility permission in System Settings.
"""

import threading
import Quartz

# Fn modifier flag
kCGEventFlagMaskSecondaryFn = 0x800000


class FnListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self._fn_down = False
        self._thread = None
        self._run_loop = None

    def _callback(self, proxy, event_type, event, refcon):
        flags = Quartz.CGEventGetFlags(event)
        fn_pressed = bool(flags & kCGEventFlagMaskSecondaryFn)

        if fn_pressed and not self._fn_down:
            self._fn_down = True
            self.on_press()
        elif not fn_pressed and self._fn_down:
            self._fn_down = False
            self.on_release()

        return event

    def _run(self):
        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
            self._callback,
            None,
        )

        if tap is None:
            print(
                "ERROR: Could not create event tap. "
                "Grant Accessibility permission in System Settings → "
                "Privacy & Security → Accessibility."
            )
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._run_loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(
            self._run_loop, source, Quartz.kCFRunLoopDefaultMode
        )
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._run_loop:
            Quartz.CFRunLoopStop(self._run_loop)
