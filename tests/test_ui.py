import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import AppKit

from fvt.contracts import ScreenFrame
from fvt.state import AppSnapshot, AppState, PermissionState, Readiness
from fvt.storage import RECORD_KEY_IDS, RECORD_KEY_MENU_LABELS
from fvt.ui import (
    HUD_HEIGHT,
    HUD_WIDTH,
    FileTranscribeWindow,
    RecordingHUD,
    SettingsWindow,
    SetupWindow,
    _button,
    _hud_frame_for_screen,
    _hud_palette,
    _paths_from_dragging,
    _set_button_state,
)
from generate_icon import generate_icon


class FakeDragging:
    def __init__(self, filenames):
        self._filenames = filenames

    def draggingPasteboard(self):
        return self

    def readObjectsForClasses_options_(self, *_args):
        return []

    def propertyListForType_(self, _type):
        return self._filenames


class FakeFileCoordinator:
    def __init__(self):
        self.paths = []
        self.file_result = ""
        self.cancelled = False
        self.copied = []

    def transcribe_audio_file(self, path):
        self.paths.append(Path(path))

    def cancel_file_transcription(self):
        self.cancelled = True

    def copy_transcript(self, text):
        self.copied.append(text)

    def copy_last_transcript(self):
        self.copied.append("last")


class FakeCoordinator:
    def __init__(self):
        self.record_key = "fn"
        self.record_key_calls = []
        self.tones = {"com.apple.mail": "Formal and complete."}
        self.writing_downloads = 0
        self.writing_cancels = 0
        self.writing_cancels = 0

    def request_microphone(self):
        pass

    def open_microphone_settings(self):
        pass

    def open_accessibility_settings(self):
        pass

    def download_model(self):
        pass

    def download_writing_model(self):
        self.writing_downloads += 1

    def cancel_writing_model(self):
        self.writing_cancels += 1

    def app_tones(self):
        return dict(self.tones)

    def set_app_tones(self, tones):
        self.tones = dict(tones)

    def set_record_key(self, key_id):
        self.record_key = key_id
        self.record_key_calls.append(key_id)

    def select_model_profile(self, profile_id):
        self.selected_profile = profile_id


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

    def test_setup_and_settings_expose_fast_accurate_picker(self):
        coordinator = FakeCoordinator()
        setup = SetupWindow.alloc().initWithCoordinator_(coordinator)
        settings = SettingsWindow.alloc().initWithCoordinator_(coordinator)
        snapshot = AppSnapshot(
            state=AppState.SETUP_REQUIRED,
            title="Speech model required",
            detail="Download the Fast model once, then transcription works locally.",
            readiness=Readiness(
                microphone=PermissionState.GRANTED,
                accessibility=PermissionState.GRANTED,
                model_profile="fast",
            ),
        )
        setup.update(snapshot)
        settings.update(snapshot)

        self.assertEqual(str(setup._model_picker.labelForSegment_(0)), "Fast")
        self.assertEqual(str(setup._model_picker.labelForSegment_(1)), "Accurate")
        self.assertEqual(int(setup._model_picker.selectedSegment()), 0)
        self.assertEqual(
            str(setup._model_button.accessibilityLabel()), "Download Model"
        )
        self.assertIn("Fast", str(setup._model_status.stringValue()))
        self.assertEqual(str(settings._picker.labelForSegment_(0)), "Fast")
        self.assertEqual(int(settings._picker.selectedSegment()), 0)
        self.assertEqual(
            str(settings._download_button.accessibilityLabel()), "Download Fast"
        )

        setup.modelProfileChanged_(setup._model_picker)
        self.assertEqual(coordinator.selected_profile, "fast")
        settings._picker.setSelectedSegment_(1)
        settings.modelProfileChanged_(settings._picker)
        self.assertEqual(coordinator.selected_profile, "accurate")
        setup._window.orderOut_(None)
        settings._window.orderOut_(None)

    def test_download_progress_updates_the_progress_bar(self):
        setup = SetupWindow.alloc().initWithCoordinator_(FakeCoordinator())
        settings = SettingsWindow.alloc().initWithCoordinator_(FakeCoordinator())
        snapshot = AppSnapshot(
            state=AppState.SETUP_REQUIRED,
            title="Downloading Fast model · 42%",
            readiness=Readiness(
                model_busy=True,
                model_profile="fast",
                model_download_progress=0.42,
            ),
        )
        setup.update(snapshot)
        settings.update(snapshot)
        self.assertFalse(setup._model_progress.isHidden())
        self.assertAlmostEqual(setup._model_progress.doubleValue(), 0.42, places=2)
        self.assertFalse(settings._progress.isHidden())
        self.assertAlmostEqual(settings._progress.doubleValue(), 0.42, places=2)
        self.assertIn("42%", str(settings._status.stringValue()))
        setup._window.orderOut_(None)
        settings._window.orderOut_(None)

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

    def test_file_window_shows_progress_then_copyable_result(self):
        coordinator = FakeFileCoordinator()
        window = FileTranscribeWindow.alloc().initWithCoordinator_(coordinator)
        ready = Readiness(
            microphone=PermissionState.GRANTED,
            accessibility=PermissionState.GRANTED,
            model_present=True,
            model_ready=True,
        )
        window.update(
            AppSnapshot(
                state=AppState.IDLE,
                title="Ready",
                detail="Hold fn and speak. Release to insert.",
                readiness=ready,
            )
        )
        self.assertTrue(window._choose_button.isEnabled())
        self.assertEqual(str(window._close_button.title()), "Close")

        window.update(
            AppSnapshot(
                state=AppState.FILE_TRANSCRIBING,
                title="Transcribing file",
                detail="Part 2 of 4 — meeting.wav stays on this Mac.",
                readiness=ready,
                file_name="meeting.wav",
                file_current=2,
                file_total=4,
            )
        )
        self.assertFalse(window._choose_button.isEnabled())
        self.assertEqual(str(window._close_button.title()), "Cancel")
        self.assertAlmostEqual(window._progress.doubleValue(), 2.0)
        window.closePressed_(None)
        self.assertTrue(coordinator.cancelled)

        coordinator.file_result = "hello from the file"
        window.update(
            AppSnapshot(
                state=AppState.IDLE,
                title="File transcribed",
                detail="Hold fn and speak. Release to insert.",
                readiness=ready,
                has_last_transcript=True,
            )
        )
        self.assertTrue(window._copy_button.isEnabled())
        self.assertTrue(window._save_button.isEnabled())
        self.assertEqual(window._result_text, "hello from the file")
        window.copyPressed_(None)
        self.assertEqual(coordinator.copied, ["hello from the file"])
        window._window.orderOut_(None)

    def test_file_drop_helper_keeps_supported_audio_paths(self):
        paths = _paths_from_dragging(FakeDragging(["/tmp/talk.m4a", "/tmp/notes.txt"]))
        self.assertEqual(paths, [Path("/tmp/talk.m4a"), Path("/tmp/notes.txt")])

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

    def test_settings_window_lists_hotkeys_and_saves_choice(self):
        coordinator = FakeCoordinator()
        settings = SettingsWindow.alloc().initWithCoordinator_(coordinator)
        settings.show()
        titles = [
            str(settings._hotkey_popup.itemAtIndex_(index).title())
            for index in range(settings._hotkey_popup.numberOfItems())
        ]
        self.assertEqual(
            titles,
            [RECORD_KEY_MENU_LABELS[key_id] for key_id in RECORD_KEY_IDS],
        )
        self.assertEqual(settings._hotkey_popup.indexOfSelectedItem(), 0)

        settings._hotkey_popup.selectItemAtIndex_(1)
        settings.hotkeyChanged_(settings._hotkey_popup)
        self.assertEqual(coordinator.record_key_calls, [RECORD_KEY_IDS[1]])

        settings.update(
            AppSnapshot(
                state=AppState.IDLE,
                title="Ready",
                writing_model_present=False,
                writing_model_ready=False,
            )
        )
        self.assertEqual(
            str(settings._writing_button.accessibilityLabel()), "Download Model"
        )
        self.assertTrue(settings._writing_button.isEnabled())
        settings.writingModelPressed_(None)
        self.assertEqual(coordinator.writing_downloads, 1)

        settings.update(
            AppSnapshot(
                state=AppState.IDLE,
                title="Ready",
                writing_model_busy=True,
            )
        )
        self.assertEqual(str(settings._writing_button.accessibilityLabel()), "Cancel")
        self.assertTrue(settings._writing_button.isEnabled())
        settings.writingModelPressed_(None)
        self.assertEqual(coordinator.writing_cancels, 1)
        settings._window.orderOut_(None)


if __name__ == "__main__":
    unittest.main()
