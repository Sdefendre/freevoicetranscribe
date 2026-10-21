"""Local audio-file loading, chunking, and transcript save helpers.

Hold-to-talk recordings are already 16 kHz mono WAV and stop at five minutes.
Files chosen from the menu can be longer and in other formats. This module
turns those files into the same 16 kHz mono PCM the local model already uses,
then splits long audio into overlapping chunks.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 1.0

# Formats Core Audio on macOS can typically decode. The picker and drop
# target use this list; the decoder still fails closed if a file cannot be read.
AUDIO_FILE_SUFFIXES = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".caf",
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".wav",
        ".wave",
    }
)

AudioConverter = Callable[[Path, Path], None]


class AudioFileError(RuntimeError):
    pass


def is_supported_audio_path(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_FILE_SUFFIXES


def suggested_transcript_name(audio_path: Path) -> str:
    return f"{audio_path.stem}.txt"


def write_canonical_wav(samples: np.ndarray, dest_dir: Path) -> Path:
    dest_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="chunk-", suffix=".wav", dir=dest_dir)
    path = Path(raw_path)
    os.close(fd)
    pcm = np.clip(np.asarray(samples) * 32768.0, -32768, 32767).astype(np.int16)
    try:
        path.chmod(0o600)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def write_transcript_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass


def join_transcripts(parts: list[str]) -> str:
    return " ".join(part.strip() for part in parts if part.strip())


def chunk_samples(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
) -> list[np.ndarray]:
    """Split mono PCM into overlapping windows.

    Whisper-style models work best on about 30 seconds of audio. A one-second
    overlap reduces words being cut in half at a chunk boundary.
    """
    if samples.ndim != 1:
        raise AudioFileError("Audio must be a single channel of samples.")
    if samples.size == 0:
        return []

    chunk_frames = max(1, int(chunk_seconds * sample_rate))
    overlap_frames = min(max(0, int(overlap_seconds * sample_rate)), chunk_frames - 1)
    hop = chunk_frames - overlap_frames
    if samples.size <= chunk_frames:
        return [np.asarray(samples, dtype=np.float32)]

    chunks: list[np.ndarray] = []
    start = 0
    while start < samples.size:
        end = min(start + chunk_frames, samples.size)
        chunks.append(np.asarray(samples[start:end], dtype=np.float32))
        if end >= samples.size:
            break
        start += hop
    return chunks


def load_mono_16k(
    path: Path,
    *,
    temp_dir: Path,
    convert: AudioConverter | None = None,
) -> np.ndarray:
    """Read an audio file as 16 kHz mono float32 PCM in the range [-1, 1]."""
    if not is_supported_audio_path(path):
        raise AudioFileError("Choose a wav, mp3, m4a, or similar audio file.")
    if not path.is_file():
        raise AudioFileError("That audio file could not be found.")

    if path.suffix.lower() in {".wav", ".wave"}:
        samples = _load_pcm_wav(path)
        if samples is not None:
            return samples

    temp_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    dest = temp_dir / f"convert-{path.stem}.wav"
    try:
        converter = convert or convert_with_afconvert
        converter(path, dest)
        samples = _load_pcm_wav(dest)
        if samples is None:
            raise AudioFileError("This audio file could not be read on this Mac.")
        return samples
    finally:
        dest.unlink(missing_ok=True)


def convert_with_afconvert(source: Path, dest: Path) -> None:
    """Use the built-in macOS converter so compressed files never leave this Mac."""
    dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                f"LEI16@{SAMPLE_RATE}",
                "-c",
                "1",
                str(source),
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioFileError("This Mac could not convert that audio file.") from exc
    if result.returncode != 0 or not dest.is_file():
        raise AudioFileError("This audio file could not be read on this Mac.")
    try:
        dest.chmod(0o600)
    except OSError:
        pass


def _load_pcm_wav(path: Path) -> np.ndarray | None:
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
                return None
            channels = source.getnchannels()
            rate = source.getframerate()
            payload = source.readframes(source.getnframes())
    except (OSError, wave.Error):
        return None

    if channels not in {1, 2} or rate <= 0:
        return None
    samples = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
    if channels == 2:
        if samples.size % 2:
            return None
        samples = samples.reshape(-1, 2).mean(axis=1)
    if rate != SAMPLE_RATE:
        samples = _resample_linear(samples, rate, SAMPLE_RATE)
    return samples


def _resample_linear(
    samples: np.ndarray, source_rate: int, dest_rate: int
) -> np.ndarray:
    if samples.size == 0 or source_rate == dest_rate:
        return samples
    dest_length = max(1, int(round(samples.size * dest_rate / source_rate)))
    source_times = np.arange(samples.size, dtype=np.float64)
    dest_times = np.linspace(0, samples.size - 1, dest_length, dtype=np.float64)
    return np.interp(dest_times, source_times, samples).astype(np.float32)
