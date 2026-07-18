"""Privacy-conscious application paths and diagnostic logging."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_DIR_NAME = "FreeVoiceTranscribe"


@dataclass(frozen=True, slots=True)
class AppPaths:
    support: Path
    cache: Path
    logs: Path
    temp: Path

    @classmethod
    def default(cls) -> AppPaths:
        home = Path.home()
        support = Path(
            os.environ.get(
                "FVT_APP_SUPPORT",
                home / "Library" / "Application Support" / APP_DIR_NAME,
            )
        ).expanduser()
        cache = Path(
            os.environ.get("FVT_CACHE_DIR", home / "Library" / "Caches" / APP_DIR_NAME)
        ).expanduser()
        logs = Path(
            os.environ.get("FVT_LOG_DIR", home / "Library" / "Logs" / APP_DIR_NAME)
        ).expanduser()
        return cls(support=support, cache=cache, logs=logs, temp=cache / "Temporary")

    @property
    def model_root(self) -> Path:
        return self.support / "mlx_models"

    def device_settings_path(self) -> Path:
        return self.support / "device-settings.json"

    def create(self) -> None:
        for directory in (self.support, self.cache, self.logs, self.temp):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                directory.chmod(0o700)
            except OSError:
                pass

    def configure_runtime_environment(self) -> None:
        """Point model/cache writes away from source and the read-only app bundle.

        lightning-whisper-mlx currently resolves models from ``./mlx_models``.
        The process therefore uses Application Support as its stable working
        directory. All application resources are resolved from absolute paths.
        """
        self.create()
        os.environ.setdefault("HF_HOME", str(self.cache / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(self.cache / "huggingface" / "hub"))
        os.environ.setdefault("XDG_CACHE_HOME", str(self.cache))
        os.chdir(self.support)

    def remove_stale_audio(self, *, older_than_seconds: int = 0) -> int:
        """Remove prior-launch crash leftovers before recording can begin."""
        removed = 0
        cutoff = time.time() - older_than_seconds
        for path in self.temp.glob("recording-*.wav"):
            try:
                if older_than_seconds <= 0 or path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
        return removed


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    fd, raw_temp = tempfile.mkstemp(suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            destination.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temp_path, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


class LastTranscriptStore:
    """Persist exactly one transcript locally; this is intentionally not history."""

    VERSION = 1
    POLICY = "last-only"
    MAX_CHARACTERS = 250_000

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def default(cls) -> LastTranscriptStore:
        return cls(AppPaths.default().support / "last-transcript.json")

    def load(self) -> str:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        if (
            payload.get("version") != self.VERSION
            or payload.get("policy") != self.POLICY
            or not isinstance(payload.get("text"), str)
        ):
            return ""
        return payload["text"][: self.MAX_CHARACTERS]

    def save(self, text: str) -> None:
        normalized = str(text).strip()
        if not normalized:
            self.clear()
            return
        if len(normalized) > self.MAX_CHARACTERS:
            normalized = normalized[: self.MAX_CHARACTERS]
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        payload = json.dumps(
            {
                "version": self.VERSION,
                "policy": self.POLICY,
                "text": normalized,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        fd, raw_temp = tempfile.mkstemp(
            prefix=".last-transcript-", suffix=".tmp", dir=self.path.parent
        )
        temp_path = Path(raw_temp)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as destination:
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temp_path, self.path)
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            temp_path.unlink(missing_ok=True)
            raise

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def configure_logging(paths: AppPaths) -> logging.Logger:
    """Create a small rotating diagnostic log that never records transcripts."""
    paths.create()
    logger = logging.getLogger("freevoicetranscribe")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    handler = RotatingFileHandler(
        paths.logs / "app.log",
        maxBytes=512 * 1024,
        backupCount=2,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return logger


class DeviceSettingsStore:
    VERSION = 1

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or AppPaths.default().device_settings_path()

    def _defaults(self) -> dict:
        return {
            "version": self.VERSION,
            "input_device_index": None,
            "input_device_name": None,
            "shortcuts": {"record": "fn", "hands_free": "fn+space", "cancel": "esc"},
            "history_enabled": False,
            "history_max_entries": 200,
            "replay_last_recording": False,
            "check_for_updates": True,
            "feedback_url": "https://github.com/Sdefendre/freevoicetranscribe/issues/new",
        }

    def load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return self._defaults()
        if not isinstance(payload, dict):
            return self._defaults()
        defaults = self._defaults()
        defaults.update(payload)
        return defaults

    def save(self, payload: dict) -> None:
        normalized = dict(self._defaults())
        normalized.update(payload)
        _save_json_atomic(self.path, normalized)
