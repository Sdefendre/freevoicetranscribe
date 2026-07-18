"""Serialized application coordinator for every user-visible state change."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .contracts import (
    AudioService,
    CapturedAudio,
    HotkeyService,
    InsertionService,
    PermissionService,
    TargetApp,
    TranscriptionService,
)
from .state import AppSnapshot, AppState, PermissionState, Readiness, StateStore
from .storage import LastTranscriptStore


class EventType(str, Enum):
    START = "start"
    REFRESH_SETUP = "refresh_setup"
    DOWNLOAD_MODEL = "download_model"
    MODEL_STATUS = "model_status"
    MODEL_READY = "model_ready"
    MODEL_FAILED = "model_failed"
    FN_PRESS = "fn_press"
    FN_RELEASE = "fn_release"
    FN_SPACE = "fn_space"
    CANCEL = "cancel"
    AUDIO_LIMIT = "audio_limit"
    TRANSCRIPTION_RESULT = "transcription_result"
    COPY_LAST = "copy_last"
    INSERT_LAST = "insert_last"
    DISMISS_ERROR = "dismiss_error"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventType
    payload: object | None = None


SnapshotObserver = Callable[[AppSnapshot], None]
AudioObserver = Callable[[np.ndarray], None]


class _DaemonWorkerPool:
    """Small daemon pool whose blocked tasks cannot prevent application exit."""

    def __init__(self, worker_count: int = 2) -> None:
        self._tasks: queue.Queue[tuple[Callable, tuple] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._closing = False
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"fvt-worker-{index + 1}",
                daemon=True,
            )
            for index in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def _worker(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            callback, args = task
            try:
                callback(*args)
            except Exception:
                # Task entry points own user-facing error reporting.
                logging.getLogger("freevoicetranscribe.coordinator").exception(
                    "Background task escaped its error boundary"
                )

    def submit(self, callback: Callable, *args) -> bool:
        with self._lock:
            if self._closing:
                return False
            self._tasks.put((callback, args))
            return True

    def shutdown(self, timeout: float) -> bool:
        with self._lock:
            if not self._closing:
                self._closing = True
                while True:
                    try:
                        self._tasks.get_nowait()
                    except queue.Empty:
                        break
                for _thread in self._threads:
                    self._tasks.put(None)
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)


class AppCoordinator:
    """Single-writer state machine coordinating platform service boundaries."""

    def __init__(
        self,
        *,
        audio: AudioService,
        transcriber: TranscriptionService,
        inserter: InsertionService,
        hotkey: HotkeyService,
        permissions: PermissionService,
        transcript_store=None,
        logger: logging.Logger | None = None,
        min_audio_seconds: float = 0.25,
        shutdown_timeout: float = 2.0,
    ) -> None:
        self.audio = audio
        self.transcriber = transcriber
        self.inserter = inserter
        self.hotkey = hotkey
        self.permissions = permissions
        self._logger = logger or logging.getLogger("freevoicetranscribe.coordinator")
        self._min_audio_seconds = min_audio_seconds
        self._shutdown_timeout = shutdown_timeout

        self._store = StateStore()
        self._events: queue.Queue[Event | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._executor = _DaemonWorkerPool(worker_count=2)
        self._lifecycle_lock = threading.RLock()
        self._started = False
        self._closing = False
        self._shutdown_requested = False
        self._shutdown_complete = threading.Event()
        self._snapshot_observer: SnapshotObserver | None = None
        self._audio_observer: AudioObserver | None = None
        self._model_busy = False
        self._hotkey_started = False
        self._hands_free = False
        self._target: TargetApp | None = None
        self._transcript_store = transcript_store or LastTranscriptStore.default()
        try:
            self._last_transcript = str(self._transcript_store.load() or "").strip()
        except Exception as exc:
            self._logger.warning("Last transcript load failed: %s", type(exc).__name__)
            self._last_transcript = ""
        self._store.update(has_last_transcript=bool(self._last_transcript))

    @property
    def snapshot(self) -> AppSnapshot:
        return self._store.snapshot

    def set_observers(
        self,
        *,
        on_snapshot: SnapshotObserver | None = None,
        on_audio_level: AudioObserver | None = None,
    ) -> None:
        self._snapshot_observer = on_snapshot
        self._audio_observer = on_audio_level
        self._publish()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closing or (self._thread and self._thread.is_alive()):
                return
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="fvt-coordinator",
                daemon=True,
            )
            self._thread.start()
            self._events.put(Event(EventType.START))

    def submit(self, kind: EventType, payload: object | None = None) -> None:
        with self._lifecycle_lock:
            if self._closing:
                return
            self._events.put(Event(kind, payload))

    def fn_press(self) -> None:
        self.submit(EventType.FN_PRESS)

    def fn_release(self) -> None:
        self.submit(EventType.FN_RELEASE)

    def fn_space(self) -> None:
        self.submit(EventType.FN_SPACE)

    def cancel_recording(self) -> None:
        self.submit(EventType.CANCEL)

    def refresh_setup(self) -> None:
        self.submit(EventType.REFRESH_SETUP)

    def download_model(self) -> None:
        self.submit(EventType.DOWNLOAD_MODEL)

    def request_microphone(self) -> None:
        self.permissions.request_microphone(
            lambda _granted: self.submit(EventType.REFRESH_SETUP)
        )

    def open_accessibility_settings(self) -> None:
        self.permissions.open_accessibility_settings()

    def open_microphone_settings(self) -> None:
        self.permissions.open_microphone_settings()

    def copy_last_transcript(self) -> None:
        self.submit(EventType.COPY_LAST)

    def insert_last_transcript(self) -> None:
        self.submit(EventType.INSERT_LAST)

    def dismiss_error(self) -> None:
        self.submit(EventType.DISMISS_ERROR)

    def _run(self) -> None:
        while True:
            event = self._events.get()
            if event is None:
                return
            try:
                self._handle(event)
            except Exception:
                self._logger.exception("Coordinator event failed: %s", event.kind.value)
                if event.kind is not EventType.SHUTDOWN:
                    self._set_error(
                        "Something went wrong",
                        "The app recovered safely. Try dictating again.",
                        action="dismiss",
                    )
                else:
                    self._shutdown_complete.set()
                    return

    def _handle(self, event: Event) -> None:
        handlers = {
            EventType.START: self._handle_start,
            EventType.REFRESH_SETUP: self._refresh_readiness,
            EventType.DOWNLOAD_MODEL: self._download_model,
            EventType.MODEL_STATUS: self._model_status,
            EventType.MODEL_READY: self._model_ready,
            EventType.MODEL_FAILED: self._model_failed,
            EventType.FN_PRESS: self._fn_press,
            EventType.FN_RELEASE: self._fn_release,
            EventType.FN_SPACE: self._fn_space,
            EventType.CANCEL: self._cancel,
            EventType.AUDIO_LIMIT: self._audio_limit,
            EventType.TRANSCRIPTION_RESULT: self._transcription_result,
            EventType.COPY_LAST: self._copy_last,
            EventType.INSERT_LAST: self._insert_last,
            EventType.DISMISS_ERROR: self._dismiss_error,
            EventType.SHUTDOWN: self._handle_shutdown,
        }
        handlers[event.kind](event.payload)

    def _handle_start(self, _payload=None) -> None:
        self._refresh_readiness()
        if self.transcriber.model_present and not self.transcriber.is_ready:
            self._prepare_model(allow_download=False)

    def _readiness(self) -> Readiness:
        return Readiness(
            microphone=self.permissions.microphone_state(),
            accessibility=self.permissions.accessibility_state(),
            model_present=self.transcriber.model_present,
            model_ready=self.transcriber.is_ready,
            model_busy=self._model_busy,
        )

    def _refresh_readiness(self, _payload=None) -> None:
        readiness = self._readiness()
        state = self._store.snapshot.state

        if self._hotkey_started and not self._hotkey_is_healthy():
            try:
                self.hotkey.stop()
            except Exception:
                self._logger.warning("Unhealthy hotkey cleanup failed", exc_info=True)
            self._hotkey_started = False

        if (
            readiness.accessibility is not PermissionState.GRANTED
            and self._hotkey_started
        ):
            try:
                self.hotkey.stop()
            except Exception:
                self._logger.warning(
                    "Hotkey permission-loss cleanup failed", exc_info=True
                )
            self._hotkey_started = False

        if state is AppState.RECORDING and (
            not readiness.permissions_ready or not self._hotkey_started
        ):
            self.hotkey.set_recording_active(False)
            try:
                self.audio.cancel()
            except Exception as exc:
                self._logger.warning(
                    "Permission-loss audio cancellation failed: %s",
                    type(exc).__name__,
                )
                self._store.update(readiness=readiness)
                self._set_error(
                    "Recording is still stopping",
                    "Access changed while recording. Quit the app if the microphone does not release.",
                    action="setup",
                )
                return
            self._hands_free = False
            self._target = None
            self._store.transition(
                AppState.IDLE,
                title="Recording cancelled",
                detail="Access changed while the microphone was active.",
                readiness=readiness,
                hands_free=False,
                target_screen=None,
            )
            state = AppState.IDLE

        if (
            readiness.accessibility is PermissionState.GRANTED
            and not self._hotkey_started
        ):
            try:
                started = bool(self.hotkey.start())
            except Exception as exc:
                self._logger.warning("Hotkey startup failed: %s", type(exc).__name__)
                started = False
            self._hotkey_started = started and self._hotkey_is_healthy()
            if started and not self._hotkey_started:
                try:
                    self.hotkey.stop()
                except Exception:
                    self._logger.warning(
                        "Failed hotkey startup cleanup failed", exc_info=True
                    )

        active = state in {
            AppState.TRANSCRIBING,
            AppState.INSERTING,
            AppState.SHUTTING_DOWN,
        }
        if active:
            self._store.update(readiness=readiness)
            self._publish()
            return

        if readiness.ready and self._hotkey_started:
            self._store.transition(
                AppState.IDLE,
                title="Ready",
                detail="Hold fn and speak. Release to insert.",
                readiness=readiness,
                error_action=None,
                hands_free=False,
            )
        elif readiness.ready:
            self._store.transition(
                AppState.SETUP_REQUIRED,
                title="Hold-to-talk shortcut unavailable",
                detail="Open Accessibility settings, then return to retry.",
                readiness=readiness,
                error_action="setup",
                hands_free=False,
            )
        else:
            title, detail, action = self._setup_message(readiness)
            self._store.transition(
                AppState.SETUP_REQUIRED,
                title=title,
                detail=detail,
                readiness=readiness,
                error_action=action,
                hands_free=False,
            )
        self._publish()

    def _hotkey_is_healthy(self) -> bool:
        try:
            health = self.hotkey.is_healthy
        except AttributeError:
            return True
        except Exception:
            return False
        try:
            return bool(health() if callable(health) else health)
        except Exception:
            return False

    def _setup_message(self, readiness: Readiness) -> tuple[str, str, str]:
        if readiness.microphone.value != "granted":
            return (
                "Microphone access needed",
                "Audio stays on this Mac and is deleted after transcription.",
                "setup",
            )
        if readiness.accessibility.value != "granted":
            return (
                "Typing access needed",
                "Accessibility lets the app detect fn and paste into your chosen app.",
                "setup",
            )
        if readiness.model_busy:
            return (
                "Preparing speech model",
                "The private local model is being prepared.",
                "setup",
            )
        if not readiness.model_present:
            return (
                "Speech model required",
                "Download the model once, then transcription works locally.",
                "setup",
            )
        return (
            "Loading speech model",
            "The local model is almost ready.",
            "setup",
        )

    def _prepare_model(self, *, allow_download: bool) -> None:
        if self._model_busy:
            return
        self._model_busy = True
        self._refresh_readiness()

        def worker() -> None:
            try:
                self.transcriber.prepare(
                    allow_download=allow_download,
                    on_status=lambda text: self.submit(EventType.MODEL_STATUS, text),
                )
                self.submit(EventType.MODEL_READY)
            except Exception as exc:
                self.submit(EventType.MODEL_FAILED, exc)

        self._executor.submit(worker)

    def _download_model(self, _payload=None) -> None:
        self._prepare_model(allow_download=True)

    def _model_status(self, payload) -> None:
        readiness = self._readiness()
        self._store.transition(
            AppState.SETUP_REQUIRED,
            title=str(payload or "Preparing speech model"),
            detail="Keep the app open. Audio is not uploaded.",
            readiness=readiness,
            error_action="setup",
        )
        self._publish()

    def _model_ready(self, _payload=None) -> None:
        self._model_busy = False
        self._refresh_readiness()

    def _model_failed(self, payload) -> None:
        self._model_busy = False
        self._logger.warning("Model setup failed: %s", type(payload).__name__)
        readiness = self._readiness()
        self._store.transition(
            AppState.SETUP_REQUIRED,
            title="Speech model could not be prepared",
            detail="Check the internet connection and available disk space, then retry.",
            readiness=readiness,
            error_action="setup",
        )
        self._publish()

    def _fn_press(self, _payload=None) -> None:
        state = self._store.snapshot.state
        if state is AppState.RECORDING:
            if self._hands_free:
                self._finish_recording()
            return
        if state in {AppState.TRANSCRIBING, AppState.INSERTING}:
            self._store.update(
                title="Still transcribing",
                detail="Your recording is being finished locally.",
            )
            self._publish()
            return
        readiness = self._readiness()
        if not readiness.ready or not self._hotkey_started:
            self._refresh_readiness()
            return
        if state is AppState.ERROR:
            self._store.transition(
                AppState.IDLE,
                title="Ready",
                detail="Hold fn and speak. Release to insert.",
                readiness=readiness,
                error_action=None,
            )
        self._begin_recording()

    def _begin_recording(self) -> None:
        target = self.inserter.capture_target()
        if target is None:
            self._set_error(
                "Choose where the text should go",
                "Focus a text field in another app, then hold fn again.",
                action="dismiss",
            )
            return
        self._target = target
        self._store.update(
            title="Opening microphone",
            detail="Audio stays on this Mac.",
            target_screen=target.screen,
        )
        self._publish()
        try:
            self.audio.start(
                on_level=self._audio_level,
                on_limit=lambda: self.submit(EventType.AUDIO_LIMIT),
            )
        except Exception as exc:
            self._logger.warning("Recording could not start: %s", type(exc).__name__)
            self._target = None
            self._set_error(
                "Couldn’t start the microphone",
                "Check Microphone access and the selected input device.",
                action="setup",
            )
            return
        self._hands_free = False
        self.hotkey.set_recording_active(True)
        self._store.transition(
            AppState.RECORDING,
            title="Listening",
            detail="Release fn to insert. Press esc to cancel.",
            hands_free=False,
            target_screen=target.screen,
            error_action=None,
        )
        self._publish()

    def _audio_level(self, samples: np.ndarray) -> None:
        observer = self._audio_observer
        if observer is not None:
            observer(samples)

    def _fn_release(self, _payload=None) -> None:
        if self._store.snapshot.state is AppState.RECORDING and not self._hands_free:
            self._finish_recording()

    def _fn_space(self, _payload=None) -> None:
        if self._store.snapshot.state is not AppState.RECORDING or self._hands_free:
            return
        self._hands_free = True
        self._store.transition(
            AppState.RECORDING,
            title="Hands-free recording",
            detail="Press fn to finish. Press esc to cancel.",
            hands_free=True,
        )
        self._publish()

    def _finish_recording(self) -> None:
        self.hotkey.set_recording_active(False)
        self._hands_free = False
        try:
            captured = self.audio.stop()
        except Exception as exc:
            self._logger.warning("Recording could not finish: %s", type(exc).__name__)
            stopping = bool(getattr(self.audio, "is_stopping", False))
            self._set_error(
                "Recording is still stopping"
                if stopping
                else "No usable audio was captured",
                "Wait a moment before trying again."
                if stopping
                else "Check the microphone and try again.",
                action="dismiss",
            )
            return
        if captured.duration_seconds < self._min_audio_seconds:
            captured.cleanup()
            self._set_error(
                "I didn’t hear enough audio",
                "Hold fn a little longer and try again.",
                action="dismiss",
            )
            return

        self._store.transition(
            AppState.TRANSCRIBING,
            title="Transcribing",
            detail="Processing locally on this Mac.",
            hands_free=False,
        )
        self._publish()
        self._executor.submit(self._transcribe_worker, captured)

    def _transcribe_worker(self, captured: CapturedAudio) -> None:
        try:
            text = self.transcriber.transcribe(captured.path)
            self.submit(EventType.TRANSCRIPTION_RESULT, (text, None))
        except Exception as exc:
            self.submit(EventType.TRANSCRIPTION_RESULT, ("", exc))
        finally:
            captured.cleanup()

    def _transcription_result(self, payload) -> None:
        text, error = payload
        if error is not None:
            self._logger.warning("Transcription failed: %s", type(error).__name__)
            self._set_error(
                "Couldn’t transcribe that recording",
                "The audio was deleted. Try again.",
                action="dismiss",
            )
            return
        text = str(text).strip()
        if not text:
            self._set_error(
                "I didn’t hear speech",
                "Try speaking closer to the microphone.",
                action="dismiss",
            )
            return
        self._last_transcript = text
        try:
            self._transcript_store.save(text)
        except Exception as exc:
            self._logger.warning("Last transcript save failed: %s", type(exc).__name__)
        self._store.transition(
            AppState.INSERTING,
            title="Inserting text",
            detail=f"Returning to {self._target.name if self._target else 'your app'}.",
            has_last_transcript=True,
        )
        self._publish()
        if self.permissions.accessibility_state() is not PermissionState.GRANTED:
            self._copy_after_insert_failure(
                text,
                "Typing access changed. The transcript was copied instead.",
            )
            return
        self._insert_text(text, self._target)

    def _insert_text(self, text: str, target: TargetApp | None) -> None:
        if target is None:
            self._copy_after_insert_failure(
                text, "The original app is no longer available."
            )
            return
        try:
            result = self.inserter.insert(text, target)
        except Exception as exc:
            self._logger.warning("Text insertion failed: %s", type(exc).__name__)
            self._copy_after_insert_failure(
                text,
                "Couldn’t type there. The transcript was copied instead.",
            )
            return
        self._target = None
        if result.warning:
            self._set_error(
                "Text inserted with a clipboard warning",
                result.warning,
                action="dismiss",
            )
            return
        self._store.transition(
            AppState.IDLE,
            title="Ready",
            detail="Hold fn and speak. Release to insert.",
            hands_free=False,
            has_last_transcript=True,
            error_action=None,
            target_screen=None,
        )
        self._publish()

    def _copy_after_insert_failure(self, text: str, detail: str) -> None:
        copied = False
        try:
            self.inserter.copy(text)
            copied = True
        except Exception as exc:
            self._logger.warning(
                "Transcript copy fallback failed: %s", type(exc).__name__
            )
        self._target = None
        self._set_error(
            "Transcript copied" if copied else "Transcript is available in the menu",
            detail
            if copied
            else "Use Copy Last Transcript from the menu to recover it.",
            action="dismiss",
        )

    def _cancel(self, _payload=None) -> None:
        if self._store.snapshot.state is not AppState.RECORDING:
            return
        self.hotkey.set_recording_active(False)
        try:
            self.audio.cancel()
        except Exception as exc:
            self._logger.warning("Audio cancellation failed: %s", type(exc).__name__)
            self._hands_free = False
            self._set_error(
                "Recording is still stopping",
                "The microphone has not released yet. Wait before trying again.",
                action="dismiss",
            )
            return
        self._hands_free = False
        self._target = None
        self._store.transition(
            AppState.IDLE,
            title="Ready",
            detail="Recording cancelled. Hold fn to try again.",
            hands_free=False,
            error_action=None,
            target_screen=None,
        )
        self._publish()

    def _audio_limit(self, _payload=None) -> None:
        if self._store.snapshot.state is AppState.RECORDING:
            self._finish_recording()

    def _copy_last(self, _payload=None) -> None:
        if not self._last_transcript:
            self._set_error(
                "No transcript yet",
                "Dictate something first, then it can be copied.",
                action="dismiss",
            )
            return
        try:
            self.inserter.copy(self._last_transcript)
        except Exception as exc:
            self._logger.warning("Copy last transcript failed: %s", type(exc).__name__)
            self._set_error(
                "Couldn’t copy the transcript",
                "The last transcript is still kept in memory.",
                action="dismiss",
            )
            return
        if self._store.snapshot.state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.INSERTING,
        }:
            self._store.update(
                detail="Last transcript copied. Current work is continuing."
            )
            self._publish()
            return
        self._return_to_ready("Copied to the clipboard")

    def _insert_last(self, _payload=None) -> None:
        if not self._last_transcript:
            self._copy_last()
            return
        if self._store.snapshot.state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.INSERTING,
        }:
            self._store.update(
                detail="Finish the current recording before inserting again."
            )
            self._publish()
            return
        if not self._readiness().permissions_ready:
            self._refresh_readiness()
            return
        target = self.inserter.capture_target()
        if target is None:
            self._set_error(
                "Choose where the text should go",
                "Focus a text field, then choose Insert Last Transcript again.",
                action="dismiss",
            )
            return
        self._target = target
        state = self._store.snapshot.state
        if state is AppState.ERROR:
            self._store.transition(AppState.IDLE)
        self._store.transition(
            AppState.INSERTING,
            title="Inserting text",
            detail=f"Returning to {target.name}.",
            target_screen=target.screen,
        )
        self._publish()
        self._insert_text(self._last_transcript, target)

    def _dismiss_error(self, _payload=None) -> None:
        if self._store.snapshot.state is AppState.ERROR:
            self._return_to_ready("Ready")
        elif self._store.snapshot.state is AppState.SETUP_REQUIRED:
            self._refresh_readiness()

    def _return_to_ready(self, title: str) -> None:
        readiness = self._readiness()
        if readiness.ready and self._hotkey_started:
            self._store.transition(
                AppState.IDLE,
                title=title,
                detail="Hold fn and speak. Release to insert.",
                readiness=readiness,
                hands_free=False,
                has_last_transcript=bool(self._last_transcript),
                error_action=None,
                target_screen=None,
            )
        else:
            self._refresh_readiness()
            return
        self._publish()

    def _set_error(self, title: str, detail: str, *, action: str) -> None:
        self.hotkey.set_recording_active(False)
        state = self._store.snapshot.state
        if state is AppState.SETUP_REQUIRED:
            # Setup failures remain on the setup surface, not behind it.
            self._store.transition(
                AppState.SETUP_REQUIRED,
                title=title,
                detail=detail,
                error_action=action,
                hands_free=False,
                has_last_transcript=bool(self._last_transcript),
            )
        else:
            self._store.transition(
                AppState.ERROR,
                title=title,
                detail=detail,
                error_action=action,
                hands_free=False,
                has_last_transcript=bool(self._last_transcript),
            )
        self._publish()

    def _publish(self) -> None:
        observer = self._snapshot_observer
        if observer is not None:
            observer(self._store.snapshot)

    def _handle_shutdown(self, _payload=None) -> None:
        self._closing = True
        self._store.transition(
            AppState.SHUTTING_DOWN,
            title="Quitting",
            detail="Cleaning up local audio.",
            hands_free=False,
        )
        self._publish()
        deadline = time.monotonic() + self._shutdown_timeout
        try:
            self.hotkey.set_recording_active(False)
        except Exception:
            self._logger.warning("Hotkey deactivation failed", exc_info=True)

        cleanup_threads = [
            self._start_cleanup("audio", self.audio.shutdown),
            self._start_cleanup("hotkey", self.hotkey.stop),
        ]
        for cleanup in cleanup_threads:
            cleanup.join(max(0.0, deadline - time.monotonic()))
        if any(cleanup.is_alive() for cleanup in cleanup_threads):
            self._logger.warning(
                "Platform cleanup exceeded the bounded shutdown window"
            )
        self._executor.shutdown(max(0.0, deadline - time.monotonic()))
        self._shutdown_complete.set()
        self._events.put(None)

    def _start_cleanup(
        self, name: str, callback: Callable[[], object]
    ) -> threading.Thread:
        def run() -> None:
            try:
                callback()
            except Exception:
                self._logger.warning(
                    "%s shutdown failed", name.capitalize(), exc_info=True
                )

        thread = threading.Thread(
            target=run,
            name=f"fvt-{name}-shutdown",
            daemon=True,
        )
        thread.start()
        return thread

    def shutdown(self, timeout: float | None = None) -> bool:
        """Request shutdown and never block the calling UI indefinitely."""
        wait_timeout = self._shutdown_timeout if timeout is None else max(0.0, timeout)
        with self._lifecycle_lock:
            if self._shutdown_complete.is_set():
                return True
            if not self._shutdown_requested:
                self._shutdown_requested = True
                self._closing = True
                if not self._started:
                    self._started = True
                    self._thread = threading.Thread(
                        target=self._run,
                        name="fvt-coordinator",
                        daemon=True,
                    )
                    self._thread.start()
                self._events.put(Event(EventType.SHUTDOWN))
        completed = self._shutdown_complete.wait(wait_timeout)
        thread = self._thread
        if completed and thread and thread is not threading.current_thread():
            thread.join(min(0.1, wait_timeout))
        return completed
