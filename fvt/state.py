"""Application state and transition validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .contracts import ScreenFrame
from .models import DEFAULT_PROFILE_ID


class AppState(StrEnum):
    BOOTING = "booting"
    SETUP_REQUIRED = "setup_required"
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    FILE_TRANSCRIBING = "file_transcribing"
    INSERTING = "inserting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class PermissionState(StrEnum):
    UNKNOWN = "unknown"
    GRANTED = "granted"
    DENIED = "denied"
    RESTRICTED = "restricted"


class DictationMode(StrEnum):
    """What happens after the local speech model produces a transcript."""

    DICTATE = "dictate"
    WRITE = "write"
    COMMAND = "command"

    @classmethod
    def parse(cls, value: object) -> DictationMode:
        try:
            return cls(str(value))
        except ValueError:
            return cls.DICTATE


def idle_detail_for_mode(mode: DictationMode, hotkey_label: str = "fn") -> str:
    if mode is DictationMode.WRITE:
        return f"Hold {hotkey_label} and speak. Release to insert cleaned-up text."
    if mode is DictationMode.COMMAND:
        return f"Select text, hold {hotkey_label}, and speak a rewrite instruction."
    if mode is DictationMode.DICTATE:
        return f"Hold {hotkey_label} and speak. Release to insert."
    unhandled: DictationMode = mode
    raise AssertionError(f"unhandled dictation mode: {unhandled}")


@dataclass(frozen=True, slots=True)
class Readiness:
    microphone: PermissionState = PermissionState.UNKNOWN
    accessibility: PermissionState = PermissionState.UNKNOWN
    model_present: bool = False
    model_ready: bool = False
    model_busy: bool = False
    model_profile: str = DEFAULT_PROFILE_ID
    model_download_progress: float | None = None

    @property
    def permissions_ready(self) -> bool:
        return (
            self.microphone is PermissionState.GRANTED
            and self.accessibility is PermissionState.GRANTED
        )

    @property
    def ready(self) -> bool:
        return self.permissions_ready and self.model_ready


@dataclass(frozen=True, slots=True)
class AppSnapshot:
    state: AppState = AppState.BOOTING
    title: str = "Starting"
    detail: str = "Checking local setup"
    readiness: Readiness = Readiness()
    hands_free: bool = False
    has_last_transcript: bool = False
    error_action: str | None = None
    target_screen: ScreenFrame | None = None
    file_name: str = ""
    file_current: int = 0
    file_total: int = 0
    dictation_mode: DictationMode = DictationMode.DICTATE
    writing_model_present: bool = False
    writing_model_ready: bool = False
    writing_model_busy: bool = False


class InvalidTransition(RuntimeError):
    pass


_ALLOWED_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.BOOTING: {
        AppState.SETUP_REQUIRED,
        AppState.IDLE,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.SETUP_REQUIRED: {
        AppState.SETUP_REQUIRED,
        AppState.IDLE,
        AppState.FILE_TRANSCRIBING,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.IDLE: {
        AppState.IDLE,
        AppState.SETUP_REQUIRED,
        AppState.RECORDING,
        AppState.FILE_TRANSCRIBING,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.RECORDING: {
        AppState.RECORDING,
        AppState.IDLE,
        AppState.TRANSCRIBING,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.TRANSCRIBING: {
        AppState.TRANSCRIBING,
        AppState.INSERTING,
        AppState.IDLE,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.FILE_TRANSCRIBING: {
        AppState.FILE_TRANSCRIBING,
        AppState.IDLE,
        AppState.SETUP_REQUIRED,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.INSERTING: {
        AppState.INSERTING,
        AppState.IDLE,
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.ERROR: {
        AppState.ERROR,
        AppState.SETUP_REQUIRED,
        AppState.IDLE,
        AppState.RECORDING,
        AppState.FILE_TRANSCRIBING,
        AppState.SHUTTING_DOWN,
    },
    AppState.SHUTTING_DOWN: {AppState.SHUTTING_DOWN},
}


class StateStore:
    """Single-writer snapshot store with explicit legal transitions."""

    def __init__(self) -> None:
        self._snapshot = AppSnapshot()

    @property
    def snapshot(self) -> AppSnapshot:
        return self._snapshot

    def transition(self, state: AppState, **changes) -> AppSnapshot:
        current = self._snapshot.state
        if state not in _ALLOWED_TRANSITIONS[current]:
            raise InvalidTransition(
                f"Illegal application transition: {current} -> {state}"
            )
        self._snapshot = replace(self._snapshot, state=state, **changes)
        return self._snapshot

    def update(self, **changes) -> AppSnapshot:
        self._snapshot = replace(self._snapshot, **changes)
        return self._snapshot
