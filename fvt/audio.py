"""Bounded microphone capture with session-local lifecycle state."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyaudio

from .contracts import AudioLevelCallback, AudioLimitCallback, CapturedAudio


class AudioError(RuntimeError):
    pass


def _default_input_device(pa: pyaudio.PyAudio) -> tuple[int | None, str | None]:
    try:
        info = pa.get_default_input_device_info()
    except (OSError, IOError):
        return None, None
    return info.get("index"), info.get("name")


@dataclass(slots=True)
class _CaptureSession:
    """All mutable state for exactly one recorder run.

    A session is retained until its worker is confirmed stopped. That prevents a
    late worker from writing into a newer recording's buffers or stop events.
    """

    on_level: AudioLevelCallback | None = None
    on_limit: AudioLimitCallback | None = None
    opened: threading.Event = field(default_factory=threading.Event)
    stop_requested: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    frames: list[bytes] = field(default_factory=list)
    frame_count: int = 0
    error: Exception | None = None
    limit_reached: bool = False
    cancelled: bool = False
    thread: threading.Thread | None = None
    stream: object | None = None
    pa: object | None = None


class AudioRecorder:
    """Capture 16 kHz mono PCM with a startup handshake and hard size bound."""

    def __init__(
        self,
        temp_dir: Path,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        frames_per_buffer: int = 1_024,
        max_seconds: int = 300,
        open_timeout: float = 4.0,
        stop_timeout: float = 5.0,
        abort_timeout: float = 0.5,
        pa_factory: Callable[[], object] | None = None,
        logger: logging.Logger | None = None,
        input_device_index: int | None = None,
    ) -> None:
        self._temp_dir = temp_dir
        self._sample_rate = sample_rate
        self._channels = channels
        self._frames_per_buffer = frames_per_buffer
        self._max_frames = sample_rate * max_seconds
        self._open_timeout = open_timeout
        self._stop_timeout = stop_timeout
        self._abort_timeout = abort_timeout
        self._pa_factory = pa_factory or pyaudio.PyAudio
        self._logger = logger or logging.getLogger("freevoicetranscribe.audio")
        self._lock = threading.RLock()
        self._session: _CaptureSession | None = None
        self._input_device_index = input_device_index
        self._input_device_name: str | None = None
        self._used_default_device = input_device_index is None

    @property
    def input_device_index(self) -> int | None:
        return self._input_device_index

    @property
    def input_device_name(self) -> str | None:
        if self._input_device_name is not None:
            return self._input_device_name
        if self._input_device_index is not None and self._session is None:
            try:
                with self._pa_factory() as pa:
                    info = pa.get_device_info_by_index(self._input_device_index)
            except Exception:
                return None
            return info.get("name") if isinstance(info, dict) else None
        return self._input_device_name or (
            self._session.pa.get_default_input_device_info().get("name")
            if self._session is not None and self._session.pa is not None
            else None
        )

    @property
    def used_default_device(self) -> bool:
        return self._used_default_device

    @property
    def is_recording(self) -> bool:
        with self._lock:
            session = self._session
            thread = session.thread if session else None
            return bool(
                session
                and thread
                and thread.is_alive()
                and session.opened.is_set()
                and not session.stop_requested.is_set()
            )

    @property
    def is_stopping(self) -> bool:
        with self._lock:
            return bool(self._session and self._session.stop_requested.is_set())

    def start(
        self,
        on_level: AudioLevelCallback | None = None,
        on_limit: AudioLimitCallback | None = None,
    ) -> None:
        """Open the input stream before reporting success to the coordinator."""
        with self._lock:
            if self._session is not None:
                raise AudioError(
                    "The previous recording has not finished cleaning up"
                    if self._session.stop_requested.is_set()
                    else "A recording is already active"
                )
            session = _CaptureSession(on_level=on_level, on_limit=on_limit)
            session.thread = threading.Thread(
                target=self._record_loop,
                args=(session,),
                name="fvt-audio-capture",
                # A broken device driver must not hold the whole app open.
                daemon=True,
            )
            self._session = session
            session.thread.start()

        if not session.opened.wait(self._open_timeout):
            session.cancelled = True
            session.stop_requested.set()
            stopped = self._join_session(session)
            if stopped:
                self._release_session(session)
            raise AudioError(
                "The microphone did not become ready in time"
                if stopped
                else "The microphone did not become ready and is still stopping"
            )

        if session.error is not None:
            self._join_session(session)
            error = session.error
            self._release_session(session)
            raise AudioError(f"Could not open the microphone: {error}") from error

    def _record_loop(self, session: _CaptureSession) -> None:
        limit_callback: AudioLimitCallback | None = None
        try:
            session.pa = self._pa_factory()
            open_kwargs = {
                "format": pyaudio.paInt16,
                "channels": self._channels,
                "rate": self._sample_rate,
                "frames_per_buffer": self._frames_per_buffer,
                "input": True,
            }
            try:
                open_kwargs["input_device_index"] = self._input_device_index
            except Exception:
                pass
            session.stream = session.pa.open(**open_kwargs)
            session.opened.set()

            while not session.stop_requested.is_set():
                remaining = self._max_frames - session.frame_count
                if remaining <= 0:
                    session.limit_reached = True
                    limit_callback = session.on_limit
                    break

                requested = min(self._frames_per_buffer, remaining)
                data = session.stream.read(requested, exception_on_overflow=False)
                frame_count = len(data) // (2 * self._channels)
                if frame_count <= 0:
                    continue
                session.frames.append(data)
                session.frame_count += frame_count
                if session.on_level is not None:
                    try:
                        session.on_level(np.frombuffer(data, dtype=np.int16))
                    except Exception:
                        self._logger.debug("Audio level callback failed", exc_info=True)
        except Exception as exc:
            session.error = exc
            self._logger.warning("Microphone capture failed: %s", type(exc).__name__)
            session.opened.set()
        finally:
            self._close_audio_resources(session)
            session.finished.set()
            if limit_callback is not None and not session.cancelled:
                try:
                    limit_callback()
                except Exception:
                    self._logger.debug("Audio limit callback failed", exc_info=True)

    @staticmethod
    def _close_audio_resources(session: _CaptureSession) -> None:
        stream, session.stream = session.stream, None
        pa, session.pa = session.pa, None
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    def stop(self) -> CapturedAudio:
        """Stop capture and persist a private WAV only after the worker exits."""
        session = self._current_session()
        session.stop_requested.set()
        if not self._join_session(session):
            # Keep the live session and stop flag intact. A second start remains
            # blocked until a later stop/cancel confirms the worker has exited.
            raise AudioError("The microphone is still stopping")

        frames = list(session.frames)
        frame_count = session.frame_count
        error = session.error
        self._release_session(session)
        if error is not None:
            raise AudioError(f"Microphone capture failed: {error}") from error
        if session.cancelled:
            raise AudioError("The recording was cancelled")
        if not frames or frame_count <= 0:
            raise AudioError("No audio was captured")

        path = self._write_wav(frames)
        return CapturedAudio(
            path=path,
            duration_seconds=frame_count / self._sample_rate,
            frame_count=frame_count,
        )

    def _current_session(self) -> _CaptureSession:
        with self._lock:
            if self._session is None:
                raise AudioError("No recording is active")
            return self._session

    def _join_session(self, session: _CaptureSession) -> bool:
        thread = session.thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return not thread.is_alive()
        thread.join(self._stop_timeout)
        if thread.is_alive():
            stream = session.stream
            if stream is not None:
                try:
                    stream.abort_stream()
                except Exception:
                    pass
            thread.join(self._abort_timeout)
        return not thread.is_alive()

    def _release_session(self, session: _CaptureSession) -> None:
        thread = session.thread
        if thread is not None and thread.is_alive():
            raise AudioError("Cannot release a live recording session")
        with self._lock:
            if self._session is session:
                self._session = None

    def _write_wav(self, frames: list[bytes]) -> Path:
        self._temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix="recording-", suffix=".wav", dir=self._temp_dir
        )
        path = Path(raw_path)
        os.close(fd)
        try:
            path.chmod(0o600)
            with wave.open(str(path), "wb") as output:
                output.setnchannels(self._channels)
                output.setsampwidth(2)
                output.setframerate(self._sample_rate)
                output.writeframes(b"".join(frames))
            return path
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise AudioError(f"Could not save the temporary recording: {exc}") from exc

    def cancel(self) -> bool:
        """Request cancellation and report success only after capture has stopped."""
        with self._lock:
            session = self._session
        if session is None:
            return True
        session.cancelled = True
        session.stop_requested.set()
        if not self._join_session(session):
            raise AudioError("Recording cancellation is still in progress")
        self._release_session(session)
        return True

    def shutdown(self) -> bool:
        return self.cancel()
