import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fvt.contracts import CapturedAudio, InsertResult, TargetApp
from fvt.coordinator import AppCoordinator
from fvt.file_audio import SAMPLE_RATE, AudioFileCancelled
from fvt.state import AppState, PermissionState


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for coordinator state")


class FakePermissions:
    def __init__(self):
        self.microphone = PermissionState.GRANTED
        self.accessibility = PermissionState.GRANTED

    def microphone_state(self):
        return self.microphone

    def accessibility_state(self):
        return self.accessibility

    def request_microphone(self, callback):
        callback(True)

    def open_accessibility_settings(self):
        pass

    def open_microphone_settings(self):
        pass


class FakeHotkey:
    last_error = None

    def __init__(self, *, healthy=True, start_succeeds=True):
        self.started = False
        self.active = False
        self.healthy = healthy
        self.start_succeeds = start_succeeds
        self.start_calls = 0
        self.stop_calls = 0
        self.record_key = "fn"

    @property
    def is_healthy(self):
        return self.started and self.healthy

    def set_record_key(self, key_id):
        self.record_key = key_id

    def start(self):
        self.start_calls += 1
        self.started = self.start_succeeds
        return self.start_succeeds

    def set_recording_active(self, active):
        self.active = active

    def stop(self):
        self.stop_calls += 1
        self.started = False


class FakeTranscriptStore:
    def __init__(self, text=""):
        self.text = text
        self.saved = []

    def load(self):
        return self.text

    def save(self, text):
        self.text = text
        self.saved.append(text)


class FakeSettingsStore:
    def __init__(self, profile="accurate", payload=None):
        if isinstance(profile, dict) and payload is None:
            payload = profile
            profile = payload.get("speech_model_profile", "accurate")
        self.profile = profile
        self.payload = payload or {
            "shortcuts": {"record": "fn", "hands_free": "fn+space", "cancel": "esc"},
            "speech_model_profile": profile,
        }
        self.payload.setdefault("speech_model_profile", profile)
        self.saved = []

    def speech_model_profile(self):
        return self.profile

    def save_speech_model_profile(self, profile_id):
        self.profile = profile_id
        self.payload["speech_model_profile"] = profile_id
        self.saved.append(profile_id)

    def load(self):
        data = dict(self.payload)
        data["speech_model_profile"] = self.profile
        return data

    def save(self, payload):
        self.payload = dict(payload)
        if "speech_model_profile" in payload:
            self.profile = payload["speech_model_profile"]
        self.saved.append(dict(payload))


class FakeTranscriber:
    def __init__(
        self,
        text="hello world",
        *,
        ready=True,
        present=True,
        model_name="distil-large-v3",
        present_by_name=None,
        error=None,
        texts=None,
    ):
        self.text = text
        self.texts = list(texts) if texts is not None else None
        self._ready = ready
        self._present = present
        self._model_name = model_name
        self.present_by_name = dict(present_by_name or {model_name: present})
        self.prepared_names = []
        self.calls = 0
        self.sample_lengths = []
        self.error = error

    @property
    def is_ready(self):
        return self._ready

    @property
    def model_present(self):
        return self._present

    @property
    def model_name(self):
        return self._model_name

    def set_model_name(self, model_name):
        self._model_name = model_name
        self._ready = False
        if model_name in self.present_by_name:
            self._present = self.present_by_name[model_name]

    def prepare(self, *, allow_download, on_status=None, on_progress=None):
        if on_status:
            on_status("Preparing speech model")
        if on_progress:
            on_progress(0.5)
        self._present = True
        self.present_by_name[self._model_name] = True
        self._ready = True
        self.prepared_names.append(self._model_name)

    def transcribe(self, _path):
        self.calls += 1
        return self._next_text()

    def transcribe_samples(self, samples):
        self.calls += 1
        self.sample_lengths.append(len(samples))
        if self.error:
            raise RuntimeError(self.error)
        return self._next_text()

    def _next_text(self):
        if self.texts:
            return self.texts[min(self.calls - 1, len(self.texts) - 1)]
        return self.text


class FakeAudio:
    def __init__(self, temp_dir):
        self.temp_dir = Path(temp_dir)
        self.started = False
        self.cancelled = False
        self.fail_cancel = False

    def start(self, on_level=None, on_limit=None):
        self.started = True

    def stop(self):
        self.started = False
        path = self.temp_dir / "recording.wav"
        path.write_bytes(b"audio")
        return CapturedAudio(path=path, duration_seconds=1.0, frame_count=16_000)

    def cancel(self):
        if self.fail_cancel:
            raise RuntimeError("device still blocking")
        self.started = False
        self.cancelled = True

    def shutdown(self):
        self.cancel()


class FakeInserter:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.inserted = []
        self.copied = []
        self.target = TargetApp(pid=123, name="Editor", bundle_id="editor.app")

    def capture_target(self):
        return self.target

    def insert(self, text, target):
        if self.fail:
            raise RuntimeError("paste denied")
        self.inserted.append((text, target))
        return InsertResult(method="paste", clipboard_restored=True)

    def copy(self, text):
        self.copied.append(text)


class CoordinatorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audio = FakeAudio(self.temp.name)
        self.transcriber = FakeTranscriber()
        self.inserter = FakeInserter()
        self.hotkey = FakeHotkey()
        self.permissions = FakePermissions()
        self.transcript_store = FakeTranscriptStore()
        self.coordinator = AppCoordinator(
            audio=self.audio,
            transcriber=self.transcriber,
            inserter=self.inserter,
            hotkey=self.hotkey,
            permissions=self.permissions,
            transcript_store=self.transcript_store,
            tap_max_hold=0,
            temp_dir=Path(self.temp.name),
        )
        self.coordinator.start()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)

    def tearDown(self):
        self.coordinator.shutdown()
        self.temp.cleanup()

    def test_hold_release_transcribes_inserts_and_cleans_temp_audio(self):
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.fn_release()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertEqual(self.inserter.inserted[0][0], "hello world")
        self.assertTrue(self.coordinator.snapshot.has_last_transcript)
        self.assertEqual(self.transcript_store.saved, ["hello world"])
        self.assertFalse((Path(self.temp.name) / "recording.wav").exists())

    def test_hands_free_ignores_release_and_fn_press_finishes(self):
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.fn_space()
        wait_for(lambda: self.coordinator.snapshot.hands_free)
        self.coordinator.fn_release()
        time.sleep(0.05)
        self.assertEqual(self.coordinator.snapshot.state, AppState.RECORDING)
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertEqual(len(self.inserter.inserted), 1)

    def test_escape_cancels_without_transcribing(self):
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.cancel_recording()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertTrue(self.audio.cancelled)
        self.assertEqual(self.transcriber.calls, 0)

    def test_cancel_failure_is_reported_honestly(self):
        self.audio.fail_cancel = True
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.cancel_recording()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.ERROR)
        self.assertEqual(self.coordinator.snapshot.title, "Recording is still stopping")
        self.assertNotIn("cancelled", self.coordinator.snapshot.detail.lower())

    def test_permission_loss_cancels_active_recording(self):
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.permissions.microphone = PermissionState.DENIED
        self.coordinator.refresh_setup()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.SETUP_REQUIRED)
        self.assertTrue(self.audio.cancelled)
        self.assertFalse(self.hotkey.active)

    def test_insert_failure_copies_and_keeps_persistent_recovery(self):
        self.inserter.fail = True
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.fn_release()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.ERROR)
        self.assertEqual(self.inserter.copied, ["hello world"])
        self.assertTrue(self.coordinator.snapshot.has_last_transcript)

    def test_double_tap_locks_and_later_tap_finishes(self):
        self.coordinator._tap_max_hold = 0.2
        self.coordinator._double_tap_window = 0.3
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.fn_release()
        time.sleep(0.02)
        self.assertEqual(self.coordinator.snapshot.state, AppState.RECORDING)
        self.assertFalse(self.coordinator.snapshot.hands_free)
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.hands_free)
        self.coordinator.fn_release()
        time.sleep(0.02)
        self.assertEqual(self.coordinator.snapshot.state, AppState.RECORDING)
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertEqual(self.inserter.inserted[0][0], "hello world")

    def test_single_tap_finishes_after_double_tap_window(self):
        self.coordinator._tap_max_hold = 0.2
        self.coordinator._double_tap_window = 0.12
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.fn_release()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertEqual(self.inserter.inserted[0][0], "hello world")

    def test_changing_hotkey_while_recording_finishes_the_take(self):
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.set_record_key("f13")
        wait_for(lambda: self.coordinator.snapshot.state is AppState.IDLE)
        self.assertEqual(self.hotkey.record_key, "f13")
        self.assertFalse(self.hotkey.active)
        self.assertEqual(self.inserter.inserted[0][0], "hello world")

    def _write_wav(self, name: str, seconds: float) -> Path:
        path = Path(self.temp.name) / name
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(b"\x00\x00" * int(seconds * SAMPLE_RATE))
        return path

    def test_file_transcription_chunks_saves_and_does_not_insert(self):
        path = self._write_wav("meeting.wav", 75)
        self.coordinator.transcribe_audio_file(path)
        wait_for(lambda: self.coordinator.file_result == "hello world")
        self.assertEqual(self.coordinator.snapshot.state, AppState.IDLE)
        self.assertEqual(self.coordinator.snapshot.title, "File transcribed")
        self.assertEqual(self.transcriber.calls, 3)
        self.assertEqual(self.inserter.inserted, [])
        self.assertTrue(self.coordinator.snapshot.has_last_transcript)
        self.assertEqual(self.transcript_store.saved, ["hello world"])

    def test_file_transcription_trims_overlapping_chunk_text(self):
        path = self._write_wav("meeting.wav", 75)
        self.transcriber.texts = [
            "and then we go",
            "we go to the store",
            "to the store later",
        ]
        self.coordinator.transcribe_audio_file(path)
        wait_for(
            lambda: self.coordinator.file_result == "and then we go to the store later"
        )
        self.assertEqual(self.transcriber.calls, 3)
        self.assertEqual(self.inserter.inserted, [])
        self.assertEqual(
            self.transcript_store.saved, ["and then we go to the store later"]
        )

    def test_file_transcription_is_blocked_while_recording(self):
        path = self._write_wav("later.wav", 1)
        self.coordinator.fn_press()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.RECORDING)
        self.coordinator.transcribe_audio_file(path)
        time.sleep(0.05)
        self.assertEqual(self.coordinator.snapshot.state, AppState.RECORDING)
        self.assertEqual(self.transcriber.calls, 0)

    def test_file_transcription_can_be_cancelled_between_chunks(self):
        path = self._write_wav("long.wav", 75)
        started = threading.Event()
        release = threading.Event()

        class SlowTranscriber(FakeTranscriber):
            def transcribe_samples(self, samples):
                self.calls += 1
                self.sample_lengths.append(len(samples))
                started.set()
                release.wait(1.0)
                return self.text

        self.transcriber = SlowTranscriber()
        self.coordinator.transcriber = self.transcriber
        self.coordinator.transcribe_audio_file(path)
        self.assertTrue(started.wait(1.0))
        self.coordinator.cancel_file_transcription()
        release.set()
        wait_for(
            lambda: self.coordinator.snapshot.title == "File transcription cancelled"
        )
        self.assertEqual(self.coordinator.file_result, "")
        self.assertLess(self.transcriber.calls, 3)

    def test_file_transcription_can_be_cancelled_during_load(self):
        path = Path(self.temp.name) / "talk.m4a"
        path.write_bytes(b"not a wav")
        started = threading.Event()

        def blocking_load(*_args, **kwargs):
            started.set()
            cancel = kwargs.get("cancel_event")
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if cancel is not None and cancel.is_set():
                    raise AudioFileCancelled("File transcription was cancelled.")
                time.sleep(0.01)
            raise AssertionError("load was not cancelled")

        with patch("fvt.coordinator.load_mono_16k", side_effect=blocking_load):
            self.coordinator.transcribe_audio_file(path)
            self.assertTrue(started.wait(1.0))
            self.coordinator.cancel_file_transcription()
            wait_for(
                lambda: (
                    self.coordinator.snapshot.title == "File transcription cancelled"
                )
            )
        self.assertEqual(self.coordinator.file_result, "")
        self.assertEqual(self.transcriber.calls, 0)

    def test_file_transcription_works_when_only_the_model_is_ready(self):
        path = self._write_wav("clip.wav", 1)
        self.permissions.microphone = PermissionState.DENIED
        self.coordinator.refresh_setup()
        wait_for(lambda: self.coordinator.snapshot.state is AppState.SETUP_REQUIRED)
        self.coordinator.transcribe_audio_file(path)
        wait_for(lambda: self.coordinator.file_result == "hello world")
        self.assertEqual(self.inserter.inserted, [])
        self.assertTrue(self.coordinator.snapshot.has_last_transcript)


class CoordinatorSetupTests(unittest.TestCase):
    def test_missing_model_is_prepared_only_after_explicit_download(self):
        with tempfile.TemporaryDirectory() as temp:
            transcriber = FakeTranscriber(ready=False, present=False)
            coordinator = AppCoordinator(
                audio=FakeAudio(temp),
                transcriber=transcriber,
                inserter=FakeInserter(),
                hotkey=FakeHotkey(),
                permissions=FakePermissions(),
                transcript_store=FakeTranscriptStore(),
                temp_dir=Path(temp),
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.SETUP_REQUIRED)
            self.assertFalse(transcriber.is_ready)
            coordinator.download_model()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertTrue(transcriber.is_ready)
            coordinator.shutdown()

    def test_file_transcription_requires_a_ready_model(self):
        with tempfile.TemporaryDirectory() as temp:
            transcriber = FakeTranscriber(ready=False, present=False)
            coordinator = AppCoordinator(
                audio=FakeAudio(temp),
                transcriber=transcriber,
                inserter=FakeInserter(),
                hotkey=FakeHotkey(),
                permissions=FakePermissions(),
                transcript_store=FakeTranscriptStore(),
                temp_dir=Path(temp),
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.SETUP_REQUIRED)
            path = Path(temp) / "clip.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(SAMPLE_RATE)
                output.writeframes(b"\x00\x00" * SAMPLE_RATE)
            coordinator.transcribe_audio_file(path)
            wait_for(lambda: coordinator.snapshot.title == "Speech model required")
            self.assertEqual(transcriber.calls, 0)
            coordinator.shutdown()

    def test_selecting_fast_without_files_does_not_auto_download(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = FakeSettingsStore()
            transcriber = FakeTranscriber(
                ready=True,
                present=True,
                present_by_name={"distil-large-v3": True, "small": False},
            )
            coordinator = AppCoordinator(
                audio=FakeAudio(temp),
                transcriber=transcriber,
                inserter=FakeInserter(),
                hotkey=FakeHotkey(),
                permissions=FakePermissions(),
                transcript_store=FakeTranscriptStore(),
                settings_store=settings,
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            coordinator.select_model_profile("fast")
            wait_for(lambda: coordinator.snapshot.state is AppState.SETUP_REQUIRED)
            self.assertEqual(transcriber.model_name, "small")
            self.assertFalse(transcriber.is_ready)
            self.assertEqual(settings.saved, ["fast"])
            self.assertEqual(coordinator.snapshot.readiness.model_profile, "fast")
            self.assertEqual(transcriber.prepared_names, [])
            coordinator.download_model()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertEqual(transcriber.prepared_names, ["small"])
            coordinator.shutdown()

    def test_switching_back_to_downloaded_accurate_prepares_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            transcriber = FakeTranscriber(
                ready=False,
                present=True,
                present_by_name={"distil-large-v3": True, "small": True},
                model_name="small",
            )
            settings = FakeSettingsStore("fast")
            coordinator = AppCoordinator(
                audio=FakeAudio(temp),
                transcriber=transcriber,
                inserter=FakeInserter(),
                hotkey=FakeHotkey(),
                permissions=FakePermissions(),
                transcript_store=FakeTranscriptStore(),
                settings_store=settings,
            )
            coordinator.start()
            wait_for(lambda: transcriber.is_ready)
            coordinator.select_model_profile("accurate")
            wait_for(lambda: coordinator.snapshot.readiness.model_profile == "accurate")
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertEqual(transcriber.model_name, "distil-large-v3")
            self.assertIn("distil-large-v3", transcriber.prepared_names)
            coordinator.shutdown()

    def test_unhealthy_hotkey_never_reports_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            hotkey = FakeHotkey(healthy=False)
            coordinator = AppCoordinator(
                audio=FakeAudio(temp),
                transcriber=FakeTranscriber(),
                inserter=FakeInserter(),
                hotkey=hotkey,
                permissions=FakePermissions(),
                transcript_store=FakeTranscriptStore(),
                temp_dir=Path(temp),
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.SETUP_REQUIRED)
            self.assertEqual(
                coordinator.snapshot.title, "Hold-to-talk shortcut unavailable"
            )
            self.assertGreaterEqual(hotkey.stop_calls, 1)
            coordinator.shutdown()


class CoordinatorLifecycleTests(unittest.TestCase):
    def _coordinator(self, temp, **overrides):
        services = {
            "audio": FakeAudio(temp),
            "transcriber": FakeTranscriber(),
            "inserter": FakeInserter(),
            "hotkey": FakeHotkey(),
            "permissions": FakePermissions(),
            "transcript_store": FakeTranscriptStore(),
            "temp_dir": Path(temp),
            "shutdown_timeout": 0.15,
            "tap_max_hold": 0,
        }
        services.update(overrides)
        return AppCoordinator(**services)

    def test_shutdown_before_start_is_idempotent_and_time_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            coordinator = self._coordinator(temp)
            started = time.monotonic()
            self.assertTrue(coordinator.shutdown(timeout=0.4))
            self.assertLess(time.monotonic() - started, 0.4)
            self.assertTrue(coordinator.shutdown(timeout=0.01))

    def test_blocked_transcription_worker_cannot_freeze_quit(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()

            class BlockedTranscriber(FakeTranscriber):
                def transcribe(self, _path):
                    entered.set()
                    release.wait(2.0)
                    return "eventually"

            coordinator = self._coordinator(temp, transcriber=BlockedTranscriber())
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            coordinator.fn_press()
            wait_for(lambda: coordinator.snapshot.state is AppState.RECORDING)
            coordinator.fn_release()
            self.assertTrue(entered.wait(1.0))

            started = time.monotonic()
            self.assertTrue(coordinator.shutdown(timeout=0.4))
            self.assertLess(time.monotonic() - started, 0.4)
            release.set()

    def test_blocked_platform_cleanup_cannot_freeze_quit(self):
        with tempfile.TemporaryDirectory() as temp:
            release = threading.Event()

            class BlockedShutdownAudio(FakeAudio):
                def shutdown(self):
                    release.wait(2.0)

            coordinator = self._coordinator(temp, audio=BlockedShutdownAudio(temp))
            started = time.monotonic()
            self.assertTrue(coordinator.shutdown(timeout=0.4))
            self.assertLess(time.monotonic() - started, 0.4)
            release.set()

    def test_persisted_last_transcript_is_available_after_relaunch(self):
        with tempfile.TemporaryDirectory() as temp:
            store = FakeTranscriptStore("from the previous launch")
            inserter = FakeInserter()
            coordinator = self._coordinator(
                temp, transcript_store=store, inserter=inserter
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertTrue(coordinator.snapshot.has_last_transcript)
            coordinator.copy_last_transcript()
            wait_for(lambda: inserter.copied == ["from the previous launch"])
            coordinator.shutdown()

    def test_accessibility_loss_before_result_never_pastes_to_stale_target(self):
        with tempfile.TemporaryDirectory() as temp:
            entered = threading.Event()
            release = threading.Event()
            permissions = FakePermissions()
            inserter = FakeInserter()

            class BlockedTranscriber(FakeTranscriber):
                def transcribe(self, _path):
                    entered.set()
                    release.wait(1.0)
                    return "safe fallback"

            coordinator = self._coordinator(
                temp,
                transcriber=BlockedTranscriber(),
                permissions=permissions,
                inserter=inserter,
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            coordinator.fn_press()
            wait_for(lambda: coordinator.snapshot.state is AppState.RECORDING)
            coordinator.fn_release()
            self.assertTrue(entered.wait(1.0))
            permissions.accessibility = PermissionState.DENIED
            coordinator.refresh_setup()
            release.set()
            wait_for(lambda: coordinator.snapshot.state is AppState.ERROR)
            self.assertEqual(inserter.inserted, [])
            self.assertEqual(inserter.copied, ["safe fallback"])
            coordinator.shutdown()

    def test_persisted_record_key_is_used_in_ready_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = FakeSettingsStore(
                {
                    "shortcuts": {
                        "record": "right-option",
                        "hands_free": "right-option+space",
                        "cancel": "esc",
                    }
                }
            )
            coordinator = self._coordinator(temp, settings_store=settings)
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertEqual(coordinator.record_key, "right-option")
            self.assertIn("right option", coordinator.snapshot.detail)
            coordinator.set_record_key("f13")
            wait_for(lambda: coordinator.record_key == "f13")
            self.assertIn("F13", coordinator.snapshot.detail)
            self.assertEqual(
                settings.payload["shortcuts"]["record"],
                "f13",
            )
            coordinator.shutdown()


if __name__ == "__main__":
    unittest.main()
