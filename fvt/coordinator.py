"""Serialized application coordinator for every user-visible state change."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from .contracts import (
    AudioService,
    CapturedAudio,
    HotkeyService,
    InsertionService,
    PermissionService,
    TargetApp,
    TextCleanupService,
    TranscriptionService,
)
from .file_audio import (
    AudioFileCancelled,
    AudioFileError,
    chunk_samples,
    join_transcripts,
    load_mono_16k,
    write_canonical_wav,
)
from .models import (
    model_name_for_profile,
    normalize_profile,
    profile_for,
    profile_for_model_name,
)
from .state import (
    AppSnapshot,
    AppState,
    DictationMode,
    PermissionState,
    Readiness,
    StateStore,
    idle_detail_for_mode,
)
from .storage import (
    DEFAULT_APP_TONES,
    AppPaths,
    LastTranscriptStore,
    normalize_record_key,
    record_key_label,
)


class EventType(StrEnum):
    START = "start"
    REFRESH_SETUP = "refresh_setup"
    DOWNLOAD_MODEL = "download_model"
    SELECT_MODEL = "select_model"
    MODEL_STATUS = "model_status"
    MODEL_READY = "model_ready"
    MODEL_FAILED = "model_failed"
    FN_PRESS = "fn_press"
    FN_RELEASE = "fn_release"
    FN_SPACE = "fn_space"
    DOUBLE_TAP_TIMEOUT = "double_tap_timeout"
    SET_RECORD_KEY = "set_record_key"
    CANCEL = "cancel"
    AUDIO_LIMIT = "audio_limit"
    TRANSCRIPTION_RESULT = "transcription_result"
    TRANSCRIBE_FILE = "transcribe_file"
    FILE_TRANSCRIPTION_PROGRESS = "file_transcription_progress"
    FILE_TRANSCRIPTION_RESULT = "file_transcription_result"
    CANCEL_FILE = "cancel_file"
    CLEANUP_STATUS = "cleanup_status"
    SET_DICTATION_MODE = "set_dictation_mode"
    SET_APP_TONES = "set_app_tones"
    DOWNLOAD_WRITING_MODEL = "download_writing_model"
    WRITING_MODEL_STATUS = "writing_model_status"
    WRITING_MODEL_READY = "writing_model_ready"
    WRITING_MODEL_FAILED = "writing_model_failed"
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


class CommandCleanupError(RuntimeError):
    """Command mode could not rewrite the selection; do not paste the instruction."""


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
        settings_store=None,
        temp_dir: Path | None = None,
        cleaner: TextCleanupService | None = None,
        logger: logging.Logger | None = None,
        min_audio_seconds: float = 0.25,
        shutdown_timeout: float = 2.0,
        double_tap_window: float = 0.35,
        tap_max_hold: float = 0.25,
    ) -> None:
        self.audio = audio
        self.transcriber = transcriber
        self.inserter = inserter
        self.hotkey = hotkey
        self.permissions = permissions
        self.settings_store = settings_store
        self.cleaner = cleaner
        self._logger = logger or logging.getLogger("freevoicetranscribe.coordinator")
        self._min_audio_seconds = min_audio_seconds
        self._shutdown_timeout = shutdown_timeout
        self._double_tap_window = double_tap_window
        self._tap_max_hold = tap_max_hold
        self._record_key = "fn"
        self._press_started_at = 0.0
        self._awaiting_double_tap = False
        self._double_tap_timer: threading.Timer | None = None
        self._temp_dir = (
            Path(temp_dir) if temp_dir is not None else AppPaths.default().temp
        )

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
        self._download_progress = None
        self._model_profile = self._load_model_profile()
        self._writing_model_busy = False
        self._dictation_mode = DictationMode.DICTATE
        self._app_tones = dict(DEFAULT_APP_TONES)
        self._selected_text = ""
        self._load_writing_preferences()
        self._hotkey_started = False
        self._hands_free = False
        self._target: TargetApp | None = None
        self._transcript_store = transcript_store or LastTranscriptStore.default()
        try:
            self._last_transcript = str(self._transcript_store.load() or "").strip()
        except Exception as exc:
            self._logger.warning("Last transcript load failed: %s", type(exc).__name__)
            self._last_transcript = ""
        self._store.update(
            has_last_transcript=bool(self._last_transcript),
            dictation_mode=self._dictation_mode,
            **self._writing_fields(),
        )
        if settings_store is not None:
            try:
                shortcuts = settings_store.load().get("shortcuts") or {}
                self._record_key = normalize_record_key(shortcuts.get("record"))
            except Exception as exc:
                self._logger.warning(
                    "Shortcut settings load failed: %s", type(exc).__name__
                )
                self._record_key = "fn"
        self._file_cancel = threading.Event()
        self._file_result = ""

    @property
    def snapshot(self) -> AppSnapshot:
        return self._store.snapshot

    @property
    def record_key(self) -> str:
        return self._record_key

    @property
    def last_transcript(self) -> str:
        return self._last_transcript

    @property
    def file_result(self) -> str:
        return self._file_result

    def set_observers(
        self,
        *,
        on_snapshot: SnapshotObserver | None = None,
        on_audio_level: AudioObserver | None = None,
        on_feedback=None,
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

    def select_model_profile(self, profile_id: str) -> None:
        self.submit(EventType.SELECT_MODEL, profile_id)

    def download_writing_model(self) -> None:
        self.submit(EventType.DOWNLOAD_WRITING_MODEL)

    def set_dictation_mode(self, mode: str) -> None:
        self.submit(EventType.SET_DICTATION_MODE, mode)

    def set_app_tones(self, tones: dict[str, str]) -> None:
        self.submit(EventType.SET_APP_TONES, tones)

    def app_tones(self) -> dict[str, str]:
        return dict(self._app_tones)

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

    def set_record_key(self, key_id: str) -> None:
        self.submit(EventType.SET_RECORD_KEY, key_id)

    def transcribe_audio_file(self, path: Path | str) -> None:
        self.submit(EventType.TRANSCRIBE_FILE, Path(path))

    def cancel_file_transcription(self) -> None:
        with self._lifecycle_lock:
            self._file_cancel.set()
        self.submit(EventType.CANCEL_FILE)

    def copy_transcript(self, text: str) -> None:
        self.submit(EventType.COPY_LAST, text)

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
            EventType.SELECT_MODEL: self._select_model,
            EventType.MODEL_STATUS: self._model_status,
            EventType.MODEL_READY: self._model_ready,
            EventType.MODEL_FAILED: self._model_failed,
            EventType.FN_PRESS: self._fn_press,
            EventType.FN_RELEASE: self._fn_release,
            EventType.FN_SPACE: self._fn_space,
            EventType.DOUBLE_TAP_TIMEOUT: self._double_tap_timeout,
            EventType.SET_RECORD_KEY: self._set_record_key,
            EventType.CANCEL: self._cancel,
            EventType.AUDIO_LIMIT: self._audio_limit,
            EventType.TRANSCRIPTION_RESULT: self._transcription_result,
            EventType.TRANSCRIBE_FILE: self._transcribe_file,
            EventType.FILE_TRANSCRIPTION_PROGRESS: self._file_transcription_progress,
            EventType.FILE_TRANSCRIPTION_RESULT: self._file_transcription_result,
            EventType.CANCEL_FILE: self._cancel_file,
            EventType.CLEANUP_STATUS: self._cleanup_status,
            EventType.SET_DICTATION_MODE: self._set_dictation_mode,
            EventType.SET_APP_TONES: self._set_app_tones,
            EventType.DOWNLOAD_WRITING_MODEL: self._download_writing_model,
            EventType.WRITING_MODEL_STATUS: self._writing_model_status,
            EventType.WRITING_MODEL_READY: self._writing_model_ready,
            EventType.WRITING_MODEL_FAILED: self._writing_model_failed,
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
        if (
            self._dictation_mode is not DictationMode.DICTATE
            and self.cleaner is not None
            and self.cleaner.model_present
            and not self.cleaner.is_ready
        ):
            self._prepare_writing_model(allow_download=False)

    def _idle_detail(self) -> str:
        return idle_detail_for_mode(self._dictation_mode, self._hotkey_label())

    def _writing_fields(self) -> dict:
        cleaner = self.cleaner
        return {
            "writing_model_present": bool(cleaner.model_present) if cleaner else False,
            "writing_model_ready": bool(cleaner.is_ready) if cleaner else False,
            "writing_model_busy": self._writing_model_busy,
        }

    def _load_writing_preferences(self) -> None:
        if self.settings_store is None:
            return
        try:
            payload = self.settings_store.load()
        except Exception as exc:
            self._logger.warning(
                "Writing preferences load failed: %s", type(exc).__name__
            )
            return
        self._dictation_mode = DictationMode.parse(payload.get("dictation_mode"))
        tones = payload.get("app_tones")
        if isinstance(tones, dict):
            self._app_tones = {
                str(bundle_id).strip(): str(prompt)
                for bundle_id, prompt in tones.items()
                if str(bundle_id).strip()
            }

    def _persist_writing_preferences(self) -> None:
        if self.settings_store is None:
            return
        try:
            payload = self.settings_store.load()
            payload["dictation_mode"] = self._dictation_mode.value
            payload["app_tones"] = dict(self._app_tones)
            self.settings_store.save(payload)
        except Exception as exc:
            self._logger.warning(
                "Writing preferences save failed: %s", type(exc).__name__
            )

    def _tone_for(self, target: TargetApp | None) -> str:
        if target is None:
            return ""
        return str(self._app_tones.get(target.bundle_id) or "").strip()

    def _read_selected_text(self, target: TargetApp | None) -> str:
        if target is None:
            return ""
        reader = getattr(self.inserter, "read_selected_text", None)
        if reader is None:
            return ""
        try:
            return str(reader(target) or "").strip()
        except Exception as exc:
            self._logger.warning("Selected text read failed: %s", type(exc).__name__)
            return ""

    def _readiness(self) -> Readiness:
        return Readiness(
            microphone=self.permissions.microphone_state(),
            accessibility=self.permissions.accessibility_state(),
            model_present=self.transcriber.model_present,
            model_ready=self.transcriber.is_ready,
            model_busy=self._model_busy,
            model_profile=self._model_profile,
            model_download_progress=self._download_progress,
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
            self._cancel_double_tap_timeout()
            self._awaiting_double_tap = False
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
            AppState.FILE_TRANSCRIBING,
            AppState.INSERTING,
            AppState.SHUTTING_DOWN,
        }
        if active:
            self._store.update(
                readiness=readiness,
                dictation_mode=self._dictation_mode,
                **self._writing_fields(),
            )
            self._publish()
            return

        if readiness.ready and self._hotkey_started:
            self._store.transition(
                AppState.IDLE,
                title="Ready",
                detail=self._ready_detail(),
                readiness=readiness,
                error_action=None,
                hands_free=False,
                dictation_mode=self._dictation_mode,
                **self._writing_fields(),
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
                "Accessibility lets the app detect the hold-to-talk shortcut and paste into your chosen app.",
                "setup",
            )
        if readiness.model_busy:
            return (
                "Preparing speech model",
                "The private local model is being prepared.",
                "setup",
            )
        if not readiness.model_present:
            label = profile_for(readiness.model_profile).label
            return (
                "Speech model required",
                f"Download the {label} model once, then transcription works locally.",
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
            last_text = "Preparing speech model"

            def on_status(text: str) -> None:
                nonlocal last_text
                last_text = text
                self.submit(EventType.MODEL_STATUS, {"text": text})

            def on_progress(fraction: float) -> None:
                self.submit(
                    EventType.MODEL_STATUS,
                    {"text": last_text, "progress": fraction},
                )

            try:
                self.transcriber.prepare(
                    allow_download=allow_download,
                    on_status=on_status,
                    on_progress=on_progress,
                )
                self.submit(EventType.MODEL_READY)
            except Exception as exc:
                self.submit(EventType.MODEL_FAILED, exc)

        self._executor.submit(worker)

    def _download_model(self, _payload=None) -> None:
        self._prepare_model(allow_download=True)

    def _select_model(self, payload) -> None:
        profile_id = normalize_profile(payload)
        state = self._store.snapshot.state
        if self._model_busy or state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.INSERTING,
        }:
            self._publish()
            return
        if profile_id == self._model_profile and self.transcriber.is_ready:
            self._refresh_readiness()
            return
        self._model_profile = profile_id
        if self.settings_store is not None:
            try:
                self.settings_store.save_speech_model_profile(profile_id)
            except Exception as exc:
                self._logger.warning(
                    "Speech model preference could not be saved: %s",
                    type(exc).__name__,
                )
        setter = getattr(self.transcriber, "set_model_name", None)
        if callable(setter):
            setter(model_name_for_profile(profile_id))
        if self.transcriber.model_present and not self.transcriber.is_ready:
            self._prepare_model(allow_download=False)
            return
        self._refresh_readiness()

    def _load_model_profile(self) -> str:
        stored = None
        if self.settings_store is not None:
            try:
                stored = self.settings_store.speech_model_profile()
            except Exception as exc:
                self._logger.warning(
                    "Speech model preference could not be loaded: %s",
                    type(exc).__name__,
                )
        if stored is None:
            current_name = getattr(self.transcriber, "model_name", None)
            profile = (
                profile_for_model_name(current_name)
                if isinstance(current_name, str)
                else None
            )
            return profile.id if profile is not None else normalize_profile(None)
        return normalize_profile(stored)

    def _download_writing_model(self, _payload=None) -> None:
        self._prepare_writing_model(allow_download=True)

    def _prepare_writing_model(self, *, allow_download: bool) -> None:
        if self.cleaner is None or self._writing_model_busy:
            return
        self._writing_model_busy = True
        self._store.update(**self._writing_fields())
        self._publish()

        def worker() -> None:
            try:
                self.cleaner.prepare(
                    allow_download=allow_download,
                    on_status=lambda text: self.submit(
                        EventType.WRITING_MODEL_STATUS, text
                    ),
                )
                self.submit(EventType.WRITING_MODEL_READY)
            except Exception as exc:
                self.submit(EventType.WRITING_MODEL_FAILED, exc)

        self._executor.submit(worker)

    def _writing_model_status(self, payload) -> None:
        if self._store.snapshot.state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.FILE_TRANSCRIBING,
            AppState.INSERTING,
            AppState.SHUTTING_DOWN,
        }:
            self._store.update(**self._writing_fields())
            self._publish()
            return
        self._store.update(
            title=str(payload or "Preparing writing model"),
            detail="Keep the app open. Text stays on this Mac.",
            **self._writing_fields(),
        )
        self._publish()

    def _writing_model_ready(self, _payload=None) -> None:
        self._writing_model_busy = False
        if self._store.snapshot.state is AppState.IDLE:
            self._store.update(
                title="Ready",
                detail=self._ready_detail(),
                **self._writing_fields(),
            )
            self._publish()
            return
        self._store.update(**self._writing_fields())
        self._publish()

    def _writing_model_failed(self, payload) -> None:
        self._writing_model_busy = False
        self._logger.warning("Writing model setup failed: %s", type(payload).__name__)
        if self._store.snapshot.state is AppState.IDLE:
            self._store.update(
                title="Ready",
                detail="Writing model could not be prepared. Dictate still inserts as spoken.",
                **self._writing_fields(),
            )
            self._publish()
            return
        self._store.update(**self._writing_fields())
        self._publish()

    def _set_dictation_mode(self, payload) -> None:
        mode = DictationMode.parse(payload)
        self._dictation_mode = mode
        self._persist_writing_preferences()
        state = self._store.snapshot.state
        if state is AppState.IDLE:
            self._store.update(
                title="Ready",
                detail=self._ready_detail(),
                dictation_mode=mode,
                **self._writing_fields(),
            )
        else:
            self._store.update(dictation_mode=mode, **self._writing_fields())
        self._publish()
        if (
            mode is not DictationMode.DICTATE
            and self.cleaner is not None
            and not self.cleaner.is_ready
        ):
            self._prepare_writing_model(allow_download=not self.cleaner.model_present)

    def _set_app_tones(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        self._app_tones = {
            str(bundle_id).strip(): str(prompt)
            for bundle_id, prompt in payload.items()
            if str(bundle_id).strip()
        }
        self._persist_writing_preferences()
        self._store.update(**self._writing_fields())
        self._publish()

    def _model_status(self, payload) -> None:
        text = "Preparing speech model"
        progress = None
        if isinstance(payload, dict):
            text = str(payload.get("text") or text)
            raw_progress = payload.get("progress")
            if isinstance(raw_progress, (int, float)):
                progress = min(1.0, max(0.0, float(raw_progress)))
        elif payload:
            text = str(payload)
        self._download_progress = progress
        if progress is not None:
            text = f"{text} · {int(progress * 100)}%"
        readiness = self._readiness()
        self._store.transition(
            AppState.SETUP_REQUIRED,
            title=text,
            detail="Keep the app open. Audio is not uploaded.",
            readiness=readiness,
            error_action="setup",
        )
        self._publish()

    def _model_ready(self, _payload=None) -> None:
        self._model_busy = False
        self._download_progress = None
        self._refresh_readiness()

    def _model_failed(self, payload) -> None:
        self._model_busy = False
        self._download_progress = None
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

    def _hotkey_label(self) -> str:
        return record_key_label(self._record_key)

    def _ready_detail(self) -> str:
        return self._idle_detail()

    def _recording_detail(self) -> str:
        return f"Release {self._hotkey_label()} to insert. Press esc to cancel."

    def _hands_free_detail(self) -> str:
        return f"Press {self._hotkey_label()} to finish. Press esc to cancel."

    def _cancel_double_tap_timeout(self) -> None:
        timer = self._double_tap_timer
        self._double_tap_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_double_tap_timeout(self, delay: float) -> None:
        self._cancel_double_tap_timeout()
        timer = threading.Timer(
            delay, lambda: self.submit(EventType.DOUBLE_TAP_TIMEOUT)
        )
        timer.daemon = True
        self._double_tap_timer = timer
        timer.start()

    def _fn_press(self, _payload=None) -> None:
        self._cancel_double_tap_timeout()
        state = self._store.snapshot.state
        if state is AppState.RECORDING:
            if self._hands_free:
                self._awaiting_double_tap = False
                self._finish_recording()
            elif self._awaiting_double_tap:
                # Second tap arrived in time: lock instead of finishing.
                self._awaiting_double_tap = False
                self._lock_recording()
            return
        self._awaiting_double_tap = False
        if state in {
            AppState.TRANSCRIBING,
            AppState.FILE_TRANSCRIBING,
            AppState.INSERTING,
        }:
            self._store.update(
                title="Still transcribing",
                detail="Your recording is being finished locally."
                if state is not AppState.FILE_TRANSCRIBING
                else "The audio file is still being transcribed locally.",
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
                detail=self._ready_detail(),
                readiness=readiness,
                error_action=None,
            )
        self._press_started_at = time.monotonic()
        self._begin_recording()

    def _begin_recording(self) -> None:
        target = self.inserter.capture_target()
        if target is None:
            self._set_error(
                "Choose where the text should go",
                f"Focus a text field in another app, then hold {self._hotkey_label()} again.",
                action="dismiss",
            )
            return
        self._target = target
        if self._dictation_mode is DictationMode.COMMAND:
            selected = self._read_selected_text(target)
            if not selected:
                self._target = None
                self._set_error(
                    "Select text to rewrite",
                    f"Highlight the text in the other app, then hold {self._hotkey_label()} and speak an instruction.",
                    action="dismiss",
                )
                return
            self._selected_text = selected
        else:
            self._selected_text = ""
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
            detail=self._recording_detail(),
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
        if self._store.snapshot.state is not AppState.RECORDING or self._hands_free:
            return
        hold = time.monotonic() - self._press_started_at
        if self._tap_max_hold > 0 and hold <= self._tap_max_hold:
            # Wait from this release so a second tap can lock recording.
            self._awaiting_double_tap = True
            self._schedule_double_tap_timeout(self._double_tap_window)
            return
        self._awaiting_double_tap = False
        self._finish_recording()

    def _fn_space(self, _payload=None) -> None:
        self._cancel_double_tap_timeout()
        self._awaiting_double_tap = False
        self._lock_recording()

    def _lock_recording(self) -> None:
        if self._store.snapshot.state is not AppState.RECORDING or self._hands_free:
            return
        self._hands_free = True
        self._store.transition(
            AppState.RECORDING,
            title="Hands-free recording",
            detail=self._hands_free_detail(),
            hands_free=True,
        )
        self._publish()

    def _double_tap_timeout(self, _payload=None) -> None:
        if (
            self._store.snapshot.state is AppState.RECORDING
            and not self._hands_free
            and self._awaiting_double_tap
        ):
            self._awaiting_double_tap = False
            self._finish_recording()

    def _set_record_key(self, payload) -> None:
        key_id = normalize_record_key(payload)
        if key_id == self._record_key:
            return
        self._record_key = key_id
        setter = getattr(self.hotkey, "set_record_key", None)
        if callable(setter):
            try:
                setter(key_id)
            except Exception:
                self._logger.warning("Hotkey update failed", exc_info=True)
        store = self.settings_store
        if store is not None:
            try:
                data = store.load()
                data["shortcuts"] = {
                    "record": key_id,
                    "hands_free": f"{key_id}+space",
                    "cancel": "esc",
                }
                store.save(data)
            except Exception:
                self._logger.warning("Shortcut settings save failed", exc_info=True)
        if self._store.snapshot.state is AppState.RECORDING:
            # The previous key's release will never arrive. Treat this as a
            # finished take so recording cannot stay stuck.
            self._finish_recording()
            return
        if self._store.snapshot.state is AppState.IDLE:
            self._store.update(detail=self._ready_detail())
            self._publish()

    def _finish_recording(self) -> None:
        self._cancel_double_tap_timeout()
        self._awaiting_double_tap = False
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
                f"Hold {self._hotkey_label()} a little longer and try again.",
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
        self._executor.submit(
            self._transcribe_worker,
            captured,
            self._dictation_mode,
            self._tone_for(self._target),
            self._selected_text,
        )

    def _transcribe_worker(
        self,
        captured: CapturedAudio,
        mode: DictationMode,
        tone: str,
        selected_text: str,
    ) -> None:
        try:
            text = self.transcriber.transcribe(captured.path)
        except Exception as exc:
            self.submit(EventType.TRANSCRIPTION_RESULT, ("", exc))
            captured.cleanup()
            return
        captured.cleanup()
        text = str(text).strip()
        if not text or mode is DictationMode.DICTATE or self.cleaner is None:
            self.submit(EventType.TRANSCRIPTION_RESULT, (text, None))
            return
        if not getattr(self.cleaner, "is_ready", False):
            self.submit(EventType.TRANSCRIPTION_RESULT, (text, None))
            return
        self.submit(EventType.CLEANUP_STATUS, self._cleanup_title(mode))
        try:
            cleaned = str(
                self.cleaner.clean(
                    text,
                    mode=mode.value,
                    tone=tone,
                    selected_text=selected_text,
                )
                or ""
            ).strip()
        except Exception as exc:
            self._logger.warning("Writing cleanup failed: %s", type(exc).__name__)
            if mode is DictationMode.COMMAND:
                self.submit(
                    EventType.TRANSCRIPTION_RESULT, (text, CommandCleanupError())
                )
                return
            self.submit(EventType.TRANSCRIPTION_RESULT, (text, None))
            return
        if mode is DictationMode.COMMAND and not cleaned:
            self.submit(EventType.TRANSCRIPTION_RESULT, (text, CommandCleanupError()))
            return
        self.submit(EventType.TRANSCRIPTION_RESULT, (cleaned or text, None))

    def _cleanup_title(self, mode: DictationMode) -> str:
        if mode is DictationMode.COMMAND:
            return "Rewriting selected text"
        if mode is DictationMode.WRITE:
            return "Cleaning up text"
        if mode is DictationMode.DICTATE:
            return "Transcribing"
        unhandled: DictationMode = mode
        raise AssertionError(f"unhandled dictation mode: {unhandled}")

    def _cleanup_status(self, payload) -> None:
        if self._store.snapshot.state is not AppState.TRANSCRIBING:
            return
        self._store.update(
            title=str(payload or "Cleaning up text"),
            detail="Processing locally on this Mac.",
        )
        self._publish()

    def _transcription_result(self, payload) -> None:
        text, error = payload
        if isinstance(error, CommandCleanupError):
            self._last_transcript = str(text).strip()
            if self._last_transcript:
                try:
                    self._transcript_store.save(self._last_transcript)
                except Exception as exc:
                    self._logger.warning(
                        "Last transcript save failed: %s", type(exc).__name__
                    )
                self._store.update(has_last_transcript=True)
            self._copy_after_insert_failure(
                text,
                "The writing model could not rewrite the selection. Your selected text was left unchanged.",
            )
            return
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
        self._selected_text = ""
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
            detail=self._ready_detail(),
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
        self._selected_text = ""
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
        self._cancel_double_tap_timeout()
        self._awaiting_double_tap = False
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
            detail=f"Recording cancelled. Hold {self._hotkey_label()} to try again.",
            hands_free=False,
            error_action=None,
            target_screen=None,
        )
        self._selected_text = ""
        self._publish()

    def _audio_limit(self, _payload=None) -> None:
        if self._store.snapshot.state is AppState.RECORDING:
            self._finish_recording()

    def _transcribe_file(self, payload) -> None:
        path = Path(payload)
        state = self._store.snapshot.state
        if state in {
            AppState.RECORDING,
            AppState.TRANSCRIBING,
            AppState.FILE_TRANSCRIBING,
            AppState.INSERTING,
            AppState.SHUTTING_DOWN,
        }:
            self._store.update(
                detail="Finish the current work before transcribing a file."
            )
            self._publish()
            return
        if not self.transcriber.is_ready:
            self._set_error(
                "Speech model required",
                "Open Setup to download the local model, then try the file again.",
                action="setup",
            )
            return

        with self._lifecycle_lock:
            self._file_cancel.clear()
        self._file_result = ""
        if state is AppState.ERROR:
            self._store.transition(AppState.IDLE)
        self._store.transition(
            AppState.FILE_TRANSCRIBING,
            title="Transcribing file",
            detail=f"Reading {path.name} locally.",
            file_name=path.name,
            file_current=0,
            file_total=0,
            error_action=None,
        )
        self._publish()
        self._executor.submit(self._file_transcribe_worker, path)

    def _file_transcribe_worker(self, path: Path) -> None:
        try:
            samples = load_mono_16k(
                path,
                temp_dir=self._temp_dir,
                cancel_event=self._file_cancel,
            )
            if self._file_cancel.is_set():
                self.submit(EventType.FILE_TRANSCRIPTION_RESULT, ("", None, True))
                return
            chunks = chunk_samples(samples)
            if not chunks:
                self.submit(
                    EventType.FILE_TRANSCRIPTION_RESULT,
                    ("", AudioFileError("The audio file was empty."), False),
                )
                return
            parts: list[str] = []
            total = len(chunks)
            for index, chunk in enumerate(chunks, start=1):
                if self._file_cancel.is_set():
                    self.submit(EventType.FILE_TRANSCRIPTION_RESULT, ("", None, True))
                    return
                self.submit(
                    EventType.FILE_TRANSCRIPTION_PROGRESS,
                    (path.name, index, total),
                )
                parts.append(self._transcribe_file_chunk(chunk))
            self.submit(
                EventType.FILE_TRANSCRIPTION_RESULT,
                (join_transcripts(parts), None, False),
            )
        except AudioFileCancelled:
            self.submit(EventType.FILE_TRANSCRIPTION_RESULT, ("", None, True))
        except Exception as exc:
            if self._file_cancel.is_set():
                self.submit(EventType.FILE_TRANSCRIPTION_RESULT, ("", None, True))
                return
            self.submit(EventType.FILE_TRANSCRIPTION_RESULT, ("", exc, False))

    def _transcribe_file_chunk(self, samples: np.ndarray) -> str:
        transcribe_samples = getattr(self.transcriber, "transcribe_samples", None)
        if callable(transcribe_samples):
            return transcribe_samples(samples)
        path = write_canonical_wav(samples, self._temp_dir)
        try:
            return self.transcriber.transcribe(path)
        finally:
            path.unlink(missing_ok=True)

    def _file_transcription_progress(self, payload) -> None:
        if self._store.snapshot.state is not AppState.FILE_TRANSCRIBING:
            return
        name, current, total = payload
        self._store.transition(
            AppState.FILE_TRANSCRIBING,
            title="Transcribing file",
            detail=f"Part {current} of {total} — {name} stays on this Mac.",
            file_name=str(name),
            file_current=int(current),
            file_total=int(total),
        )
        self._publish()

    def _file_transcription_result(self, payload) -> None:
        text, error, cancelled = payload
        if self._store.snapshot.state is not AppState.FILE_TRANSCRIBING:
            return
        self._store.update(file_name="", file_current=0, file_total=0)
        if cancelled:
            self._file_result = ""
            self._finish_file_job("File transcription cancelled")
            return
        if error is not None:
            self._file_result = ""
            self._logger.warning("File transcription failed: %s", type(error).__name__)
            detail = (
                str(error)
                if isinstance(error, AudioFileError)
                else "The original file was not changed. Try another file."
            )
            self._set_error(
                "Couldn’t transcribe that file",
                detail,
                action="dismiss",
            )
            return
        text = str(text).strip()
        if not text:
            self._file_result = ""
            self._set_error(
                "I didn’t hear speech in that file",
                "The original file was not changed.",
                action="dismiss",
            )
            return
        self._file_result = text
        self._last_transcript = text
        try:
            self._transcript_store.save(text)
        except Exception as exc:
            self._logger.warning("Last transcript save failed: %s", type(exc).__name__)
        self._finish_file_job("File transcribed")

    def _cancel_file(self, _payload=None) -> None:
        with self._lifecycle_lock:
            self._file_cancel.set()
        if self._store.snapshot.state is AppState.FILE_TRANSCRIBING:
            self._store.update(
                title="Cancelling file transcription",
                detail="Waiting for the current part to finish.",
            )
            self._publish()

    def _finish_file_job(self, title: str) -> None:
        if self._store.snapshot.state is not AppState.FILE_TRANSCRIBING:
            return
        readiness = self._readiness()
        next_state = (
            AppState.IDLE
            if readiness.ready and self._hotkey_started
            else AppState.SETUP_REQUIRED
        )
        self._store.transition(
            next_state,
            title=title,
            detail=(
                self._ready_detail()
                if next_state is AppState.IDLE
                else "The transcript is ready in the file window."
            ),
            readiness=readiness,
            hands_free=False,
            has_last_transcript=bool(self._last_transcript),
            error_action=None if next_state is AppState.IDLE else "setup",
            target_screen=None,
            file_name="",
            file_current=0,
            file_total=0,
        )
        self._publish()
        if next_state is AppState.SETUP_REQUIRED:
            self._refresh_readiness()

    def _copy_last(self, payload=None) -> None:
        text = str(payload).strip() if payload else self._last_transcript
        if not text:
            self._set_error(
                "No transcript yet",
                "Dictate something first, then it can be copied.",
                action="dismiss",
            )
            return
        try:
            self.inserter.copy(text)
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
            AppState.FILE_TRANSCRIBING,
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
            AppState.FILE_TRANSCRIBING,
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
                detail=self._ready_detail(),
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
        self._cancel_double_tap_timeout()
        self._awaiting_double_tap = False
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
        self._cancel_double_tap_timeout()
        self._awaiting_double_tap = False
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
