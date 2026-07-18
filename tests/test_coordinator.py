import tempfile
import threading
import time
import unittest
from pathlib import Path

from fvt.contracts import CapturedAudio, InsertResult, TargetApp
from fvt.coordinator import AppCoordinator
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

    @property
    def is_healthy(self):
        return self.started and self.healthy

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


class FakeTranscriber:
    def __init__(self, text="hello world", *, ready=True, present=True):
        self.text = text
        self._ready = ready
        self._present = present
        self.calls = 0

    @property
    def is_ready(self):
        return self._ready

    @property
    def model_present(self):
        return self._present

    def prepare(self, *, allow_download, on_status=None):
        if on_status:
            on_status("Preparing speech model")
        self._present = True
        self._ready = True

    def transcribe(self, _path):
        self.calls += 1
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
            )
            coordinator.start()
            wait_for(lambda: coordinator.snapshot.state is AppState.SETUP_REQUIRED)
            self.assertFalse(transcriber.is_ready)
            coordinator.download_model()
            wait_for(lambda: coordinator.snapshot.state is AppState.IDLE)
            self.assertTrue(transcriber.is_ready)
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
            "shutdown_timeout": 0.15,
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


if __name__ == "__main__":
    unittest.main()
