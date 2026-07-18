"""Application state and transition validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from .contracts import ScreenFrame


class AppState(str, Enum):
    BOOTING = "booting"
    SETUP_REQUIRED = "setup_required"
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    INSERTING = "inserting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class PermissionState(str, Enum):
    UNKNOWN = "unknown"
    GRANTED = "granted"
    DENIED = "denied"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class Readiness:
    microphone: PermissionState = PermissionState.UNKNOWN
    accessibility: PermissionState = PermissionState.UNKNOWN
    model_present: bool = False
    model_ready: bool = False
    model_busy: bool = False

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
        AppState.ERROR,
        AppState.SHUTTING_DOWN,
    },
    AppState.IDLE: {
        AppState.IDLE,
        AppState.SETUP_REQUIRED,
        AppState.RECORDING,
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
