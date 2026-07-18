import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import AppKit

from fvt.contracts import ScreenFrame
from fvt.state import AppSnapshot, AppState, PermissionState, Readiness
from fvt.ui import (
    HUD_HEIGHT,
    HUD_WIDTH,
    RecordingHUD,
    SetupWindow,
    _button,
    _hud_frame_for_screen,
    _hud_palette,
    _set_button_state,
)
from generate_icon import generate_icon


class FakeCoordinator:
    def request_microphone(self):
        pass

    def open_microphone_settings(self):
        pass

    def open_accessibility_settings(self):
        pass

    def download_model(self):
        pass


class NativeUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AppKit.NSApplication.sharedApplication()

    def test_dynamic_button_state_keeps_voiceover_label_in_sync(self):
        button = _button(AppKit.NSMakeRect(0, 0, 140, 32), "Allow", None, b"")
        _set_button_state(
            button,
            "Download Model",
            enabled=False,
            help_text="One-time download required",
        )
        self.assertEqual(str(button.title()), "Download Model")
        self.assertEqual(str(button.accessibilityLabel()), "Download Model")
        self.assertEqual(str(button.accessibilityHelp()), "One-time download required")
        self.assertFalse(button.isEnabled())

    def test_setup_surfaces_snapshot_failure_and_gates_start_button(self):
        setup = SetupWindow.alloc().initWithCoordinator_(FakeCoordinator())
        ready = Readiness(
            microphone=PermissionState.GRANTED,
            accessibility=PermissionState.GRANTED,
            model_present=True,
            model_ready=True,
        )
        failure = AppSnapshot(
            state=AppState.SETUP_REQUIRED,
            title="Hold-to-talk shortcut unavailable",
            detail="Open Accessibility settings, then return to retry.",
            readiness=ready,
            error_action="setup",
        )
        setup.update(failure)

        self.assertEqual(
            str(setup._message_title.stringValue()),
            "Hold-to-talk shortcut unavailable",
        )
        self.assertEqual(
            str(setup._message_detail.stringValue()),
            "Open Accessibility settings, then return to retry.",
        )
        self.assertFalse(setup._done_button.isEnabled())
        self.assertEqual(str(setup._mic_button.accessibilityLabel()), "Allowed")
        self.assertEqual(str(setup._model_button.accessibilityLabel()), "Ready")

        setup.update(
            AppSnapshot(
                state=AppState.IDLE,
                title="Ready",
                detail="Hold fn and speak. Release to insert.",
                readiness=ready,
            )
        )
        self.assertTrue(setup._done_button.isEnabled())
        setup.show()
        self.assertTrue(setup._window.isVisible())
        setup._window.orderOut_(None)

    def test_hud_frame_is_clamped_inside_negative_and_narrow_displays(self):
        frames = (
            ScreenFrame(x=-1440, y=0, width=1440, height=900),
            ScreenFrame(x=300, y=-180, width=320, height=160),
        )
        for screen in frames:
            frame = _hud_frame_for_screen(screen)
            self.assertGreaterEqual(frame.origin.x, screen.x)
            self.assertGreaterEqual(frame.origin.y, screen.y)
            self.assertLessEqual(
                frame.origin.x + frame.size.width, screen.x + screen.width
            )
            self.assertLessEqual(
                frame.origin.y + frame.size.height, screen.y + screen.height
            )
            self.assertLessEqual(frame.size.width, HUD_WIDTH)
            self.assertLessEqual(frame.size.height, HUD_HEIGHT)

    def test_hud_announces_each_visible_state_change_once(self):
        hud = RecordingHUD.alloc().initWithCallbacks_(
            {"setup": lambda: None, "dismiss": lambda: None}
        )
        screen = ScreenFrame(x=0, y=0, width=1280, height=800)
        recording = AppSnapshot(
            state=AppState.RECORDING,
            title="Listening",
            detail="Release fn to insert. Press esc to cancel.",
            target_screen=screen,
        )
        transcribing = AppSnapshot(
            state=AppState.TRANSCRIBING,
            title="Transcribing",
            detail="Processing locally on this Mac.",
            target_screen=screen,
        )
        with patch("fvt.ui._post_accessibility_announcement") as announce:
            hud.update(recording)
            hud.update(recording)
            hud.update(transcribing)

        self.assertEqual(announce.call_count, 2)
        self.assertTrue(announce.call_args_list[0].kwargs["high_priority"])
        self.assertFalse(announce.call_args_list[1].kwargs["high_priority"])
        hud._hide()

    def test_hud_palette_respects_transparency_and_contrast_preferences(self):
        standard = _hud_palette(
            dark=False,
            reduce_transparency=False,
            increase_contrast=False,
        )
        accessible = _hud_palette(
            dark=True,
            reduce_transparency=True,
            increase_contrast=True,
        )
        self.assertLess(standard["background"].alphaComponent(), 1.0)
        self.assertEqual(accessible["background"].alphaComponent(), 1.0)
        self.assertGreater(accessible["border_width"], standard["border_width"])
        self.assertIs(accessible["detail"], accessible["title"])

    def test_generated_icon_is_monochrome_and_not_microphone_shaped(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "icon.png"
            generate_icon(512, str(path))
            representation = AppKit.NSBitmapImageRep.imageRepWithContentsOfFile_(
                str(path)
            )
            self.assertEqual(representation.pixelsWide(), 512)
            self.assertEqual(representation.pixelsHigh(), 512)

            background = representation.colorAtX_y_(100, 100)
            mark = representation.colorAtX_y_(197, 256)
            for color in (background, mark):
                red, green, blue, _alpha = color.getRed_green_blue_alpha_(
                    None, None, None, None
                )
                self.assertAlmostEqual(red, green, places=2)
                self.assertAlmostEqual(green, blue, places=2)


if __name__ == "__main__":
    unittest.main()
