"""Local MLX transcription service and model readiness boundary."""

from __future__ import annotations

import json
import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .contracts import StatusCallback
from .storage import AppPaths

DEFAULT_MODEL = "distil-large-v3"


class ModelUnavailable(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


class _LocalModelHandle:
    """Readiness marker for a validated on-disk model.

    The LightningWhisperMLX constructor can resolve Hugging Face repositories.
    We deliberately do not instantiate it for offline preparation.
    """

    pass


class MLXTranscriber:
    """Own the local Whisper model and serialize inference calls."""

    def __init__(
        self,
        paths: AppPaths,
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 12,
        model_factory: Callable[..., object] | None = None,
        transcribe_callable: Callable[..., dict] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._paths = paths
        self._model_name = model_name
        self._batch_size = batch_size
        self._model_factory = model_factory
        self._transcribe_callable = transcribe_callable
        self._logger = logger or logging.getLogger("freevoicetranscribe.transcription")
        self._model = None
        self._factory_backed = False
        self._prepare_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_dir(self) -> Path:
        return self._paths.model_root / self._model_name

    @property
    def model_present(self) -> bool:
        weights = self.model_dir / "weights.npz"
        config = self.model_dir / "config.json"
        try:
            return weights.stat().st_size > 0 and config.stat().st_size > 0
        except OSError:
            return False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def prepare(
        self, *, allow_download: bool, on_status: StatusCallback | None = None
    ) -> None:
        if self._model is not None:
            return
        if not allow_download and not self.model_present:
            raise ModelUnavailable("The local speech model has not been downloaded")

        with self._prepare_lock:
            if self._model is not None:
                return
            if on_status is not None:
                on_status(
                    "Loading speech model"
                    if self.model_present
                    else "Downloading speech model"
                )
            try:
                if not allow_download:
                    # Validation and the inference path both use explicit local
                    # paths. Do not construct a Hub-aware object in offline mode.
                    self._validate_local_model()
                    self._model = _LocalModelHandle()
                    self._factory_backed = False
                else:
                    factory = self._model_factory
                    if factory is None:
                        from lightning_whisper_mlx import LightningWhisperMLX

                        factory = LightningWhisperMLX
                    # AppPaths.configure_runtime_environment makes the library's
                    # hard-coded ./mlx_models location resolve to Application Support.
                    self._model = factory(
                        model=self._model_name,
                        batch_size=self._batch_size,
                    )
                    # The default model constructor downloads files; inference
                    # must still use PCM to avoid an external ffmpeg dependency.
                    self._factory_backed = self._model_factory is not None
            except Exception as exc:
                self._logger.warning(
                    "Speech model preparation failed: %s", type(exc).__name__
                )
                raise ModelUnavailable(
                    "The speech model could not be prepared. Check the network and free disk space."
                ) from exc
            if on_status is not None:
                on_status("Speech model ready")

    def _validate_local_model(self) -> None:
        if not self.model_present:
            raise ModelUnavailable("The local speech model is incomplete")
        try:
            config = json.loads((self.model_dir / "config.json").read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ModelUnavailable(
                "The local speech model configuration is invalid"
            ) from exc
        if not isinstance(config, dict):
            raise ModelUnavailable("The local speech model configuration is invalid")

    def transcribe(self, audio_path: Path) -> str:
        model = self._model
        if model is None:
            raise ModelUnavailable("The speech model is not ready")
        try:
            with self._inference_lock:
                if self._transcribe_callable is not None:
                    result = self._transcribe_callable(
                        self._read_pcm_wav(audio_path),
                        path_or_hf_repo=str(self.model_dir),
                        batch_size=self._batch_size,
                    )
                elif self._factory_backed:
                    # Test/custom model factories own their transcription adapter.
                    result = model.transcribe(str(audio_path))
                else:
                    # lightning-whisper-mlx normally shells out to ffmpeg for paths.
                    # Our recorder already produces canonical 16 kHz mono PCM, so
                    # pass normalized samples directly and keep the app self-contained.
                    from lightning_whisper_mlx.lightning import transcribe_audio

                    result = transcribe_audio(
                        self._read_pcm_wav(audio_path),
                        path_or_hf_repo=str(self.model_dir),
                        batch_size=self._batch_size,
                    )
            if not isinstance(result, dict):
                raise TypeError("Unexpected transcription response")
            return str(result.get("text", "")).strip()
        except ModelUnavailable:
            raise
        except Exception as exc:
            self._logger.warning("Local transcription failed: %s", type(exc).__name__)
            raise TranscriptionError("Local transcription failed. Try again.") from exc

    @staticmethod
    def _read_pcm_wav(audio_path: Path) -> np.ndarray:
        with wave.open(str(audio_path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != 16_000
                or source.getcomptype() != "NONE"
            ):
                raise TranscriptionError(
                    "The temporary recording had an unexpected audio format."
                )
            payload = source.readframes(source.getnframes())
        return np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
