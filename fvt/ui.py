"""Small native setup, menu-bar, and recording HUD surfaces."""

from __future__ import annotations

import math
from collections.abc import Callable

import AppKit
import numpy as np
import objc
import rumps
from Foundation import NSAttributedString, NSObject, NSTimer

from .contracts import ScreenFrame
from .state import AppSnapshot, AppState, PermissionState

HUD_WIDTH = 424
HUD_HEIGHT = 82
HUD_SCREEN_MARGIN = 12
HUD_BOTTOM_OFFSET = 72


def _label(frame, text: str, *, size: float = 13, bold: bool = False):
    field = AppKit.NSTextField.alloc().initWithFrame_(frame)
    field.setStringValue_(text)
    field.setBezeled_(False)
    field.setDrawsBackground_(False)
    field.setEditable_(False)
    field.setSelectable_(False)
    weight = AppKit.NSFontWeightSemibold if bold else AppKit.NSFontWeightRegular
    field.setFont_(AppKit.NSFont.systemFontOfSize_weight_(size, weight))
    return field


def _set_label_text(field, text: str, *, accessibility_label: str | None = None):
    """Keep visible and assistive text synchronized for dynamic labels."""
    changed = str(field.stringValue()) != text
    field.setStringValue_(text)
    field.setAccessibilityLabel_(accessibility_label or text)
    if changed:
        try:
            AppKit.NSAccessibilityPostNotification(
                field, AppKit.NSAccessibilityValueChangedNotification
            )
        except Exception:
            pass


def _set_button_state(
    button,
    title: str,
    *,
    enabled: bool,
    help_text: str | None = None,
) -> None:
    """Update every user-visible and VoiceOver-facing button property together."""
    changed = str(button.title()) != title
    button.setTitle_(title)
    button.setEnabled_(enabled)
    button.setAccessibilityLabel_(title)
    button.setAccessibilityHelp_(help_text or title)
    if changed:
        try:
            AppKit.NSAccessibilityPostNotification(
                button, AppKit.NSAccessibilityTitleChangedNotification
            )
        except Exception:
            pass


def _button(frame, title: str, target, action: bytes):
    button = AppKit.NSButton.alloc().initWithFrame_(frame)
    button.setBezelStyle_(AppKit.NSBezelStyleRounded)
    button.setTarget_(target)
    button.setAction_(action)
    _set_button_state(button, title, enabled=True)
    return button


def _display_option(selector: str) -> bool:
    workspace = AppKit.NSWorkspace.sharedWorkspace()
    method = getattr(workspace, selector, None)
    if method is None:
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _is_dark_appearance(view=None) -> bool:
    try:
        appearance = (
            view.effectiveAppearance()
            if view is not None
            else AppKit.NSAppearance.currentDrawingAppearance()
        )
        match = appearance.bestMatchFromAppearancesWithNames_(
            [AppKit.NSAppearanceNameAqua, AppKit.NSAppearanceNameDarkAqua]
        )
        return str(match) == str(AppKit.NSAppearanceNameDarkAqua)
    except Exception:
        return False


def _hud_palette(*, dark: bool, reduce_transparency: bool, increase_contrast: bool):
    """Return an adaptive, deliberately neutral HUD palette."""
    alpha = 1.0 if reduce_transparency else 0.96
    background = AppKit.NSColor.colorWithCalibratedWhite_alpha_(
        0.12 if dark else 0.96, alpha
    )
    title = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.94 if dark else 0.12, 1.0)
    detail = (
        title
        if increase_contrast
        else AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.72 if dark else 0.34, 1.0)
    )
    border = AppKit.NSColor.colorWithCalibratedWhite_alpha_(
        0.92 if dark else 0.14, 0.76 if increase_contrast else 0.18
    )
    return {
        "background": background,
        "border": border,
        "border_width": 2.0 if increase_contrast else 1.0,
        "title": title,
        "detail": detail,
        "neutral_mark": detail,
    }


def _hud_frame_for_screen(screen: ScreenFrame):
    """Fit the HUD fully inside a supplied visible display frame."""
    width = max(1.0, float(screen.width))
    height = max(1.0, float(screen.height))
    margin_x = min(float(HUD_SCREEN_MARGIN), width / 2)
    margin_y = min(float(HUD_SCREEN_MARGIN), height / 2)
    panel_width = min(float(HUD_WIDTH), max(1.0, width - 2 * margin_x))
    panel_height = min(float(HUD_HEIGHT), max(1.0, height - 2 * margin_y))

    min_x = float(screen.x) + margin_x
    max_x = float(screen.x) + width - panel_width - margin_x
    centered_x = float(screen.x) + (width - panel_width) / 2
    x = min(max(centered_x, min_x), max_x) if max_x >= min_x else centered_x

    min_y = float(screen.y) + margin_y
    max_y = float(screen.y) + height - panel_height - margin_y
    preferred_y = float(screen.y) + HUD_BOTTOM_OFFSET
    y = min(max(preferred_y, min_y), max_y) if max_y >= min_y else min_y
    return AppKit.NSMakeRect(x, y, panel_width, panel_height)


def _post_accessibility_announcement(view, text: str, *, high_priority: bool) -> None:
    try:
        AppKit.NSAccessibilityPostNotificationWithUserInfo(
            view,
            AppKit.NSAccessibilityAnnouncementRequestedNotification,
            {
                AppKit.NSAccessibilityAnnouncementKey: text,
                AppKit.NSAccessibilityPriorityKey: (
                    AppKit.NSAccessibilityPriorityHigh
                    if high_priority
                    else AppKit.NSAccessibilityPriorityMedium
                ),
            },
        )
    except Exception:
        # Assistive notifications should never block dictation.
        pass


class HUDView(AppKit.NSView):
    def initWithFrame_(self, frame):
        self = objc.super(HUDView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.snapshot = AppSnapshot()
        self.level = 0.0
        self.reduce_transparency = _display_option(
            "accessibilityDisplayShouldReduceTransparency"
        )
        self.increase_contrast = _display_option(
            "accessibilityDisplayShouldIncreaseContrast"
        )
        self.setAccessibilityElement_(True)
        self.setAccessibilityRole_(AppKit.NSAccessibilityGroupRole)
        return self

    def setSnapshot_(self, snapshot: AppSnapshot) -> None:
        self.snapshot = snapshot
        self.setAccessibilityLabel_(f"{snapshot.title}. {snapshot.detail}")
        self.setNeedsDisplay_(True)

    def drawRect_(self, rect) -> None:
        snapshot = self.snapshot
        palette = _hud_palette(
            dark=_is_dark_appearance(self),
            reduce_transparency=self.reduce_transparency,
            increase_contrast=self.increase_contrast,
        )
        palette["background"].setFill()
        path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            self.bounds(), 11, 11
        )
        path.fill()
        palette["border"].setStroke()
        path.setLineWidth_(palette["border_width"])
        path.stroke()

        bounds = self.bounds()
        center_y = bounds.size.height / 2
        self._draw_state_mark(snapshot, palette, center_y)

        text_x = 46
        right_reserve = 102 if snapshot.state is AppState.ERROR else 94
        if snapshot.state not in {AppState.ERROR, AppState.RECORDING}:
            right_reserve = 18
        text_width = max(80, bounds.size.width - text_x - right_reserve)

        title_attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_weight_(
                14, AppKit.NSFontWeightSemibold
            ),
            AppKit.NSForegroundColorAttributeName: palette["title"],
        }
        detail_attrs = {
            AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(11.5),
            AppKit.NSForegroundColorAttributeName: palette["detail"],
        }
        NSAttributedString.alloc().initWithString_attributes_(
            snapshot.title, title_attrs
        ).drawInRect_(
            AppKit.NSMakeRect(text_x, bounds.size.height - 34, text_width, 20)
        )
        NSAttributedString.alloc().initWithString_attributes_(
            snapshot.detail, detail_attrs
        ).drawInRect_(AppKit.NSMakeRect(text_x, 12, text_width, 30))

        if snapshot.state is AppState.RECORDING:
            self._draw_meter(AppKit.NSColor.systemRedColor())

    def viewDidChangeEffectiveAppearance(self) -> None:
        objc.super(HUDView, self).viewDidChangeEffectiveAppearance()
        self.setNeedsDisplay_(True)

    @objc.python_method
    def _draw_state_mark(self, snapshot, palette, center_y: float) -> None:
        center_x = 23
        if snapshot.state is AppState.RECORDING:
            color = AppKit.NSColor.systemRedColor()
            color.setFill()
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                AppKit.NSMakeRect(center_x - 6, center_y - 6, 12, 12)
            ).fill()
            return

        if snapshot.state is AppState.ERROR:
            color = AppKit.NSColor.systemOrangeColor()
            color.setStroke()
            ring = AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                AppKit.NSMakeRect(center_x - 7, center_y - 7, 14, 14)
            )
            ring.setLineWidth_(2)
            ring.stroke()
            color.setFill()
            AppKit.NSBezierPath.fillRect_(
                AppKit.NSMakeRect(center_x - 1, center_y - 3, 2, 7)
            )
            AppKit.NSBezierPath.bezierPathWithOvalInRect_(
                AppKit.NSMakeRect(center_x - 1, center_y - 7, 2, 2)
            ).fill()
            return

        # A restrained insertion caret represents local processing and insertion.
        palette["neutral_mark"].setFill()
        AppKit.NSBezierPath.fillRect_(
            AppKit.NSMakeRect(center_x - 1, center_y - 9, 2, 18)
        )
        AppKit.NSBezierPath.fillRect_(
            AppKit.NSMakeRect(center_x - 4, center_y + 8, 8, 1.5)
        )
        AppKit.NSBezierPath.fillRect_(
            AppKit.NSMakeRect(center_x - 4, center_y - 9.5, 8, 1.5)
        )

    @objc.python_method
    def _draw_meter(self, color) -> None:
        color.setFill()
        base_x = self.bounds().size.width - 72
        center_y = self.bounds().size.height / 2
        for index in range(6):
            variation = 0.55 + 0.45 * math.sin(index * 1.7)
            height = 5 + min(1.0, self.level * 8) * 28 * variation
            rect = AppKit.NSMakeRect(
                base_x + index * 9,
                center_y - height / 2,
                4,
                height,
            )
            AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                rect, 2, 2
            ).fill()


class RecordingHUD(NSObject):
    def initWithCallbacks_(self, callbacks: dict[str, Callable[[], None]]):
        self = objc.super(RecordingHUD, self).init()
        if self is None:
            return None
        self._callbacks = callbacks
        self._panel = None
        self._view = None
        self._action_button = None
        self._timer = None
        self._latest_level = 0.0
        self._snapshot = AppSnapshot()
        self._last_announcement = None
        self._observing_display_options = False
        return self

    @objc.python_method
    def _ensure_panel(self) -> None:
        if self._panel is not None:
            return
        frame = AppKit.NSMakeRect(0, 0, HUD_WIDTH, HUD_HEIGHT)
        self._panel = (
            AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                frame,
                AppKit.NSWindowStyleMaskBorderless
                | AppKit.NSWindowStyleMaskNonactivatingPanel,
                AppKit.NSBackingStoreBuffered,
                False,
            )
        )
        self._panel.setLevel_(AppKit.NSFloatingWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        self._panel.setHasShadow_(True)
        self._panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        container = AppKit.NSView.alloc().initWithFrame_(frame)
        self._view = HUDView.alloc().initWithFrame_(frame)
        container.addSubview_(self._view)
        self._action_button = _button(
            AppKit.NSMakeRect(354, 25, 72, 30),
            "Dismiss",
            self,
            b"actionPressed:",
        )
        self._action_button.setHidden_(True)
        container.addSubview_(self._action_button)
        self._panel.setContentView_(container)
        self._observe_display_options()

    @objc.python_method
    def update(self, snapshot: AppSnapshot) -> None:
        self._ensure_panel()
        self._snapshot = snapshot
        self._view.setSnapshot_(snapshot)
        visible = snapshot.state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.INSERTING,
            AppState.ERROR,
        }
        if not visible:
            self._hide()
            self._last_announcement = None
            return

        is_error = snapshot.state is AppState.ERROR
        self._action_button.setHidden_(not is_error)
        if is_error:
            title = "Setup" if snapshot.error_action == "setup" else "Dismiss"
            _set_button_state(
                self._action_button,
                title,
                enabled=True,
                help_text=(
                    "Open setup to resolve this problem."
                    if snapshot.error_action == "setup"
                    else "Dismiss this message."
                ),
            )
        self._panel.setIgnoresMouseEvents_(not is_error)
        self._position(snapshot)
        self._sync_display_options()
        self._panel.orderFrontRegardless()
        self._ensure_timer(snapshot.state is AppState.RECORDING)
        self._announce(snapshot)

    @objc.python_method
    def _position(self, snapshot: AppSnapshot) -> None:
        screen = snapshot.target_screen
        if screen is None:
            native_screen = AppKit.NSScreen.mainScreen()
            if native_screen is None:
                return
            native_frame = native_screen.visibleFrame()
            screen = ScreenFrame(
                x=float(native_frame.origin.x),
                y=float(native_frame.origin.y),
                width=float(native_frame.size.width),
                height=float(native_frame.size.height),
            )
        frame = _hud_frame_for_screen(screen)
        self._panel.setFrame_display_(frame, True)
        self._layout_contents(frame.size.width, frame.size.height)

    @objc.python_method
    def _layout_contents(self, width: float, height: float) -> None:
        content = self._panel.contentView()
        content.setFrame_(AppKit.NSMakeRect(0, 0, width, height))
        self._view.setFrame_(AppKit.NSMakeRect(0, 0, width, height))
        button_width = min(76, max(58, width * 0.22))
        self._action_button.setFrame_(
            AppKit.NSMakeRect(
                max(8, width - button_width - 14),
                max(4, (height - 30) / 2),
                button_width,
                min(30, max(22, height - 8)),
            )
        )

    @objc.python_method
    def _observe_display_options(self) -> None:
        if self._observing_display_options:
            return
        center = AppKit.NSWorkspace.sharedWorkspace().notificationCenter()
        center.addObserver_selector_name_object_(
            self,
            b"displayOptionsChanged:",
            AppKit.NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification,
            None,
        )
        self._observing_display_options = True

    def displayOptionsChanged_(self, _notification) -> None:
        self._sync_display_options()

    @objc.python_method
    def _sync_display_options(self) -> None:
        if self._panel is None:
            return
        reduce_transparency = _display_option(
            "accessibilityDisplayShouldReduceTransparency"
        )
        increase_contrast = _display_option(
            "accessibilityDisplayShouldIncreaseContrast"
        )
        self._panel.setHasShadow_(not reduce_transparency)
        if self._view is not None:
            self._view.reduce_transparency = reduce_transparency
            self._view.increase_contrast = increase_contrast
            self._view.setNeedsDisplay_(True)

    @objc.python_method
    def _announce(self, snapshot: AppSnapshot) -> None:
        announcement = f"{snapshot.title}. {snapshot.detail}".strip()
        if announcement == self._last_announcement:
            return
        self._last_announcement = announcement
        _post_accessibility_announcement(
            self._view,
            announcement,
            high_priority=snapshot.state in {AppState.RECORDING, AppState.ERROR},
        )

    @objc.python_method
    def update_audio(self, samples: np.ndarray) -> None:
        if samples is None or len(samples) == 0:
            return
        data = samples.astype(np.float32) / 32768.0
        self._latest_level = float(np.sqrt(np.mean(np.square(data))))

    @objc.python_method
    def _ensure_timer(self, needed: bool) -> None:
        if needed and self._timer is None:
            self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0 / 24.0, self, b"timerFired:", None, True
            )
        elif not needed and self._timer is not None:
            self._timer.invalidate()
            self._timer = None

    def timerFired_(self, _timer) -> None:
        if self._view is not None:
            self._view.level = self._latest_level
            self._view.setNeedsDisplay_(True)

    @objc.python_method
    def _hide(self) -> None:
        self._ensure_timer(False)
        if self._panel is not None:
            self._panel.orderOut_(None)

    def actionPressed_(self, _sender) -> None:
        if self._snapshot.error_action == "setup":
            self._callbacks["setup"]()
        else:
            self._callbacks["dismiss"]()


class SetupWindow(NSObject):
    def initWithCoordinator_(self, coordinator):
        self = objc.super(SetupWindow, self).init()
        if self is None:
            return None
        self._coordinator = coordinator
        self._window = None
        self._message_title = None
        self._message_detail = None
        self._mic_status = None
        self._access_status = None
        self._model_status = None
        self._mic_button = None
        self._access_button = None
        self._model_button = None
        self._done_button = None
        self._snapshot = AppSnapshot()
        return self

    @objc.python_method
    def _ensure_window(self) -> None:
        if self._window is not None:
            return
        rect = AppKit.NSMakeRect(0, 0, 540, 416)
        self._window = (
            AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect,
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskMiniaturizable,
                AppKit.NSBackingStoreBuffered,
                False,
            )
        )
        self._window.setTitle_("FreeVoiceTranscribe Setup")
        self._window.setReleasedWhenClosed_(False)
        self._window.center()
        content = self._window.contentView()

        title = _label(
            AppKit.NSMakeRect(28, 362, 484, 28),
            "Speak anywhere. Your audio stays on this Mac.",
            size=19,
            bold=True,
        )
        subtitle = _label(
            AppKit.NSMakeRect(28, 320, 484, 36),
            "Complete these three checks once. Recordings are deleted after local transcription.",
            size=12,
        )
        subtitle.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
        content.addSubview_(title)
        content.addSubview_(subtitle)

        self._message_title = _label(
            AppKit.NSMakeRect(28, 278, 484, 22), "Checking local setup", bold=True
        )
        self._message_detail = _label(
            AppKit.NSMakeRect(28, 244, 484, 34),
            "The app will show the next required step here.",
            size=12,
        )
        self._message_detail.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
        separator = AppKit.NSBox.alloc().initWithFrame_(
            AppKit.NSMakeRect(28, 238, 484, 1)
        )
        separator.setBoxType_(AppKit.NSBoxSeparator)
        content.addSubview_(self._message_title)
        content.addSubview_(self._message_detail)
        content.addSubview_(separator)

        self._mic_status, self._mic_button = self._add_row(
            content, 188, "Microphone", b"microphonePressed:"
        )
        self._access_status, self._access_button = self._add_row(
            content, 126, "Type in other apps", b"accessibilityPressed:"
        )
        self._model_status, self._model_button = self._add_row(
            content, 64, "Private speech model", b"modelPressed:"
        )
        self._done_button = _button(
            AppKit.NSMakeRect(376, 16, 136, 32),
            "Start Dictating",
            self,
            b"donePressed:",
        )
        self._done_button.setKeyEquivalent_("\r")
        self._window.setDefaultButtonCell_(self._done_button.cell())
        content.addSubview_(self._done_button)

    @objc.python_method
    def _add_row(self, content, y: float, name: str, action: bytes):
        name_label = _label(AppKit.NSMakeRect(28, y + 23, 210, 22), name, bold=True)
        status = _label(AppKit.NSMakeRect(28, y, 300, 22), "Checking…", size=12)
        button = _button(AppKit.NSMakeRect(376, y + 4, 136, 32), "Allow", self, action)
        content.addSubview_(name_label)
        content.addSubview_(status)
        content.addSubview_(button)
        return status, button

    @objc.python_method
    def show(self) -> None:
        self._ensure_window()
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        self._window.makeKeyAndOrderFront_(None)

    @objc.python_method
    def update(self, snapshot: AppSnapshot) -> None:
        self._ensure_window()
        self._snapshot = snapshot
        readiness = snapshot.readiness
        _set_label_text(
            self._message_title,
            snapshot.title,
            accessibility_label=f"Current status: {snapshot.title}",
        )
        _set_label_text(
            self._message_detail,
            snapshot.detail,
            accessibility_label=f"Current status detail: {snapshot.detail}",
        )
        self._set_permission_row(
            self._mic_status,
            self._mic_button,
            readiness.microphone,
            name="Microphone",
            unknown_action="Allow",
        )
        self._set_permission_row(
            self._access_status,
            self._access_button,
            readiness.accessibility,
            name="Type in other apps",
            unknown_action="Open Settings",
        )

        if readiness.model_ready:
            model_status = "Ready — runs locally"
            _set_label_text(
                self._model_status,
                model_status,
                accessibility_label=f"Private speech model status: {model_status}",
            )
            _set_button_state(
                self._model_button,
                "Ready",
                enabled=False,
                help_text=model_status,
            )
        elif readiness.model_busy:
            model_status = "Preparing… keep the app open"
            _set_label_text(
                self._model_status,
                model_status,
                accessibility_label=f"Private speech model status: {model_status}",
            )
            _set_button_state(
                self._model_button,
                "Preparing…",
                enabled=False,
                help_text=model_status,
            )
        else:
            model_status = (
                "Downloaded but not loaded"
                if readiness.model_present
                else "One-time download required"
            )
            _set_label_text(
                self._model_status,
                model_status,
                accessibility_label=f"Private speech model status: {model_status}",
            )
            _set_button_state(
                self._model_button,
                "Load Model" if readiness.model_present else "Download Model",
                enabled=True,
                help_text=model_status,
            )

        can_start = readiness.ready and snapshot.state is AppState.IDLE
        _set_button_state(
            self._done_button,
            "Start Dictating",
            enabled=can_start,
            help_text=(
                "Close setup and begin using hold-to-talk dictation."
                if can_start
                else snapshot.detail
            ),
        )

    @objc.python_method
    def _set_permission_row(
        self,
        status_label,
        button,
        state: PermissionState,
        *,
        name: str,
        unknown_action: str,
    ) -> None:
        if state is PermissionState.GRANTED:
            status = "Allowed"
            action = "Allowed"
            enabled = False
        elif state is PermissionState.RESTRICTED:
            status = "Restricted by this Mac"
            action = "Open Settings"
            enabled = True
        elif state is PermissionState.DENIED:
            status = "Not allowed"
            action = "Open Settings"
            enabled = True
        else:
            status = "Permission required"
            action = unknown_action
            enabled = True
        _set_label_text(
            status_label,
            status,
            accessibility_label=f"{name} status: {status}",
        )
        _set_button_state(
            button,
            action,
            enabled=enabled,
            help_text=f"{name}: {status}",
        )

    def microphonePressed_(self, _sender) -> None:
        if self._snapshot.readiness.microphone is PermissionState.UNKNOWN:
            self._coordinator.request_microphone()
        else:
            self._coordinator.open_microphone_settings()

    def accessibilityPressed_(self, _sender) -> None:
        self._coordinator.open_accessibility_settings()

    def modelPressed_(self, _sender) -> None:
        self._coordinator.download_model()

    def donePressed_(self, _sender) -> None:
        self._window.orderOut_(None)


class StatusBarApp(rumps.App):
    def __init__(self, coordinator):
        super().__init__(
            "FreeVoiceTranscribe",
            title="FVT",
            quit_button=None,
        )
        self._coordinator = coordinator
        self._snapshot = AppSnapshot()
        self._status_item = rumps.MenuItem("Starting", callback=None)
        self._status_item.set_callback(None)
        self._instruction_item = rumps.MenuItem(
            "Hold fn and speak. Release to insert.", callback=None
        )
        self._copy_item = rumps.MenuItem(
            "Copy Last Transcript", callback=self._copy_last
        )
        self._insert_item = rumps.MenuItem(
            "Insert Last Transcript", callback=self._insert_last
        )
        self._setup_item = rumps.MenuItem("Open Setup…", callback=self._open_setup)
        self._dismiss_item = rumps.MenuItem(
            "Dismiss Error", callback=self._dismiss_error
        )
        self._feedback_item = rumps.MenuItem(
            "Send Feedback", callback=self.open_feedback
        )
        self._quit_item = rumps.MenuItem("Quit", callback=self._quit)
        self.menu = [
            self._status_item,
            self._instruction_item,
            None,
            self._copy_item,
            self._insert_item,
            None,
            self._setup_item,
            self._dismiss_item,
            self._feedback_item,
            None,
            self._quit_item,
        ]
        callbacks = {"setup": self.show_setup, "dismiss": coordinator.dismiss_error}
        self.hud = RecordingHUD.alloc().initWithCallbacks_(callbacks)
        self.setup_window = SetupWindow.alloc().initWithCoordinator_(coordinator)
        self._did_auto_show_setup = False
        self._refresh_timer = rumps.Timer(self._refresh_setup, 1.0)

    def update(self, snapshot: AppSnapshot) -> None:
        self._snapshot = snapshot
        self._status_item.title = snapshot.title
        self._instruction_item.title = snapshot.detail
        self._copy_item.set_callback(
            self._copy_last if snapshot.has_last_transcript else None
        )
        self._insert_item.set_callback(
            self._insert_last if snapshot.has_last_transcript else None
        )
        self._dismiss_item.hidden = snapshot.state is not AppState.ERROR
        self.setup_window.update(snapshot)
        self.hud.update(snapshot)
        if snapshot.state is AppState.SETUP_REQUIRED and not self._did_auto_show_setup:
            self._did_auto_show_setup = True
            self.show_setup()

    def show_setup(self) -> None:
        self.setup_window.show()

    def _refresh_setup(self, _timer) -> None:
        self._coordinator.refresh_setup()

    def _copy_last(self, _sender) -> None:
        self._coordinator.copy_last_transcript()

    def _insert_last(self, _sender) -> None:
        self._coordinator.insert_last_transcript()

    def _open_setup(self, _sender) -> None:
        self.show_setup()

    def _dismiss_error(self, _sender) -> None:
        self._coordinator.dismiss_error()

    def _quit(self, _sender) -> None:
        self._refresh_timer.stop()
        self._coordinator.shutdown()
        rumps.quit_application()

    def open_feedback(self, _sender=None) -> None:
        settings = getattr(self._coordinator, "settings_store", None)
        url = None
        if settings is not None:
            try:
                url = settings.load().get("feedback_url")
            except Exception:
                url = None
        target = url or "https://github.com/Sdefendre/freevoicetranscribe/issues/new"
        try:
            AppKit.NSWorkspace.sharedWorkspace().openURL_(
                AppKit.NSURL.URLWithString_(target)
            )
        except Exception:
            pass

    def run(self, **options) -> None:
        self._refresh_timer.start()
        super().run(**options)
