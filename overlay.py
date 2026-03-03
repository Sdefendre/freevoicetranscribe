"""
Floating waveform overlay — Wispr Flow-style pill bar at the bottom of the screen.

Uses PyObjC (AppKit + Core Graphics) to draw a dark translucent pill with a
live audio waveform. Must be driven from the main thread.
"""

import math
import objc
import AppKit
import Quartz
from Foundation import NSObject, NSTimer, NSRunLoop, NSDefaultRunLoopMode
import numpy as np

OVERLAY_WIDTH = 400
OVERLAY_HEIGHT = 60
CORNER_RADIUS = 30
BAR_COUNT = 40
BAR_GAP = 2
BG_ALPHA = 0.75
WAVEFORM_COLOR = (0.35, 0.78, 0.98, 1.0)  # light blue
TEXT_COLOR = (1.0, 1.0, 1.0, 0.9)


class WaveformView(AppKit.NSView):
    """Custom NSView that draws audio waveform bars."""

    def initWithFrame_(self, frame):
        self = objc.super(WaveformView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._levels = np.zeros(BAR_COUNT, dtype=np.float32)
        self._label = "Listening..."
        return self

    def setLevels_(self, levels):
        self._levels = levels
        self.setNeedsDisplay_(True)

    def setLabel_(self, text):
        self._label = text
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        ctx = AppKit.NSGraphicsContext.currentContext().CGContext()

        # Dark pill background
        Quartz.CGContextSetRGBFillColor(ctx, 0.1, 0.1, 0.12, BG_ALPHA)
        bg = Quartz.CGRectMake(0, 0, rect.size.width, rect.size.height)
        path = Quartz.CGPathCreateWithRoundedRect(
            bg, CORNER_RADIUS, CORNER_RADIUS, None
        )
        Quartz.CGContextAddPath(ctx, path)
        Quartz.CGContextFillPath(ctx)

        # Label
        attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(
                12, AppKit.NSFontWeightMedium
            ),
            AppKit.NSForegroundColorAttributeName: AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                *TEXT_COLOR
            ),
        }
        label = AppKit.NSAttributedString.alloc().initWithString_attributes_(
            self._label, attrs
        )
        label_size = label.size()
        label_x = 16
        label_y = (rect.size.height - label_size.height) / 2
        label.drawAtPoint_(AppKit.NSMakePoint(label_x, label_y))

        # Waveform bars
        bar_area_x = label_x + label_size.width + 12
        bar_area_w = rect.size.width - bar_area_x - 16
        if bar_area_w <= 0:
            return

        bar_w = max(2, (bar_area_w - BAR_GAP * (BAR_COUNT - 1)) / BAR_COUNT)
        max_bar_h = rect.size.height * 0.65
        center_y = rect.size.height / 2

        Quartz.CGContextSetRGBFillColor(ctx, *WAVEFORM_COLOR)

        for i in range(BAR_COUNT):
            level = float(self._levels[i]) if i < len(self._levels) else 0.0
            h = max(3, level * max_bar_h)
            x = bar_area_x + i * (bar_w + BAR_GAP)
            y = center_y - h / 2
            bar_rect = Quartz.CGRectMake(x, y, bar_w, h)
            bar_path = Quartz.CGPathCreateWithRoundedRect(
                bar_rect, bar_w / 2, bar_w / 2, None
            )
            Quartz.CGContextAddPath(ctx, bar_path)
            Quartz.CGContextFillPath(ctx)

    def isOpaque(self):
        return False


class Overlay:
    """Manages the floating overlay panel and waveform animation."""

    def __init__(self):
        self._panel = None
        self._view = None
        self._timer = None
        self._audio_buffer = np.zeros(BAR_COUNT, dtype=np.float32)
        self._visible = False

    def _ensure_panel(self):
        if self._panel is not None:
            return

        screen = AppKit.NSScreen.mainScreen()
        sx = screen.frame().size.width
        sy = screen.frame().size.height
        x = (sx - OVERLAY_WIDTH) / 2
        y = 80  # above dock

        frame = AppKit.NSMakeRect(x, y, OVERLAY_WIDTH, OVERLAY_HEIGHT)

        self._panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(AppKit.NSFloatingWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        self._panel.setHasShadow_(True)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content_frame = AppKit.NSMakeRect(0, 0, OVERLAY_WIDTH, OVERLAY_HEIGHT)
        self._view = WaveformView.alloc().initWithFrame_(content_frame)
        self._panel.setContentView_(self._view)

    def update_audio(self, chunk):
        """Feed a numpy audio chunk (int16 or float32). Called from any thread."""
        if chunk is None or len(chunk) == 0:
            return
        # Convert to float32 if needed
        if chunk.dtype == np.int16:
            data = chunk.astype(np.float32) / 32768.0
        else:
            data = chunk.astype(np.float32)

        # Compute RMS per segment to create BAR_COUNT levels
        seg_len = max(1, len(data) // BAR_COUNT)
        levels = np.zeros(BAR_COUNT, dtype=np.float32)
        for i in range(BAR_COUNT):
            start = i * seg_len
            end = min(start + seg_len, len(data))
            if start < len(data):
                seg = data[start:end]
                rms = math.sqrt(np.mean(seg ** 2))
                levels[i] = min(1.0, rms * 8)  # amplify for visual effect

        self._audio_buffer = levels

    def _refresh(self):
        """Timer callback — update waveform on main thread."""
        if self._view is not None:
            self._view.setLevels_(self._audio_buffer)

    def show(self):
        """Show the overlay (call from main thread)."""
        self._ensure_panel()
        self._view.setLabel_("Listening...")
        self._panel.setAlphaValue_(0.0)
        self._panel.orderFront_(None)
        self._visible = True

        # Fade in
        AppKit.NSAnimationContext.beginGrouping()
        AppKit.NSAnimationContext.currentContext().setDuration_(0.2)
        self._panel.animator().setAlphaValue_(1.0)
        AppKit.NSAnimationContext.endGrouping()

        # Start refresh timer (30 fps)
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            1.0 / 30.0, self, objc.selector(self._on_timer_, signature=b"v@:@"), None, True
        )
        NSRunLoop.currentRunLoop().addTimer_forMode_(self._timer, NSDefaultRunLoopMode)

    @objc.python_method
    def _on_timer_impl(self):
        self._refresh()

    def _on_timer_(self, timer):
        self._on_timer_impl()

    def set_transcribing(self):
        """Switch label to 'Transcribing...' (call from main thread)."""
        if self._view is not None:
            self._view.setLabel_("Transcribing...")
            # Flatten waveform
            self._audio_buffer = np.full(BAR_COUNT, 0.15, dtype=np.float32)
            self._refresh()

    def hide(self):
        """Fade out and hide (call from main thread)."""
        if not self._visible:
            return
        self._visible = False

        if self._timer:
            self._timer.invalidate()
            self._timer = None

        if self._panel:
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.25)
            self._panel.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()

            # Actually hide after animation
            def _do_hide():
                if self._panel and not self._visible:
                    self._panel.orderOut_(None)

            self._perform_after(0.3, _do_hide)

    def _perform_after(self, delay, fn):
        """Schedule a function on the main thread after a delay."""
        NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            delay,
            self,
            objc.selector(self._delayed_callback_, signature=b"v@:@"),
            {"fn": fn},
            False,
        )

    def _delayed_callback_(self, timer):
        info = timer.userInfo()
        if info and "fn" in info:
            info["fn"]()
