"""Typed boundaries shared by the coordinator and platform services."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class TargetApp:
    pid: int
    name: str
    bundle_id: str
    screen: ScreenFrame | None = None


@dataclass(slots=True)
class CapturedAudio:
    path: Path
    duration_seconds: float
    frame_count: int

    def cleanup(self) -> None:
        """Delete the temporary recording. Safe to call more than once."""
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            # Cleanup is retried by the storage sweep on the next launch.
            pass


@dataclass(frozen=True, slots=True)
class InsertResult:
    method: str
    clipboard_restored: bool
    warning: str | None = None


AudioLevelCallback = Callable[[np.ndarray], None]
AudioLimitCallback = Callable[[], None]
StatusCallback = Callable[[str], None]


class AudioService(Protocol):
    def start(
        self,
        on_level: AudioLevelCallback | None = None,
        on_limit: AudioLimitCallback | None = None,
    ) -> None: ...

    def stop(self) -> CapturedAudio: ...

    def cancel(self) -> None: ...

    def shutdown(self) -> None: ...


class TranscriptionService(Protocol):
    @property
    def is_ready(self) -> bool: ...

    @property
    def model_present(self) -> bool: ...

    def prepare(
        self, *, allow_download: bool, on_status: StatusCallback | None = None
    ) -> None: ...

    def transcribe(self, audio_path: Path) -> str: ...


class InsertionService(Protocol):
    def capture_target(self) -> TargetApp | None: ...

    def insert(self, text: str, target: TargetApp) -> InsertResult: ...

    def copy(self, text: str) -> None: ...


class HotkeyService(Protocol):
    last_error: str | None

    def start(self) -> bool: ...

    def set_recording_active(self, active: bool) -> None: ...

    def stop(self) -> None: ...


class PermissionService(Protocol):
    def accessibility_state(self): ...

    def microphone_state(self): ...

    def request_microphone(self, callback: Callable[[bool], None]) -> None: ...

    def open_accessibility_settings(self) -> None: ...

    def open_microphone_settings(self) -> None: ...
