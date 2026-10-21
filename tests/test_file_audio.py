import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from fvt.file_audio import (
    SAMPLE_RATE,
    AudioFileError,
    chunk_samples,
    is_supported_audio_path,
    join_transcripts,
    load_mono_16k,
    suggested_transcript_name,
    write_canonical_wav,
    write_transcript_text,
)


def write_wav(
    path: Path, *, seconds: float, rate: int = SAMPLE_RATE, channels: int = 1
):
    frames = int(seconds * rate)
    if channels == 1:
        payload = np.zeros(frames, dtype=np.int16)
    else:
        payload = np.zeros(frames * channels, dtype=np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(payload.tobytes())


class FileAudioTests(unittest.TestCase):
    def test_supported_suffixes_include_common_audio_types(self):
        self.assertTrue(is_supported_audio_path(Path("note.m4a")))
        self.assertTrue(is_supported_audio_path(Path("Talk.MP3")))
        self.assertTrue(is_supported_audio_path(Path("clip.wav")))
        self.assertFalse(is_supported_audio_path(Path("notes.txt")))
        self.assertEqual(
            suggested_transcript_name(Path("Interview.m4a")), "Interview.txt"
        )

    def test_canonical_wav_loads_without_a_converter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "voice.wav"
            write_wav(path, seconds=0.5)
            samples = load_mono_16k(path, temp_dir=root / "tmp")
            self.assertEqual(samples.shape, (8_000,))

    def test_stereo_wav_is_mixed_and_resampled_to_16k(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "stereo.wav"
            write_wav(path, seconds=1.0, rate=44_100, channels=2)
            samples = load_mono_16k(path, temp_dir=root / "tmp")
            self.assertEqual(samples.ndim, 1)
            self.assertAlmostEqual(len(samples) / SAMPLE_RATE, 1.0, places=2)

    def test_compressed_suffix_uses_injected_converter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "clip.m4a"
            source.write_bytes(b"not a wav")
            calls = []

            def convert(src: Path, dest: Path) -> None:
                calls.append((src, dest))
                write_wav(dest, seconds=0.25)

            samples = load_mono_16k(source, temp_dir=root / "tmp", convert=convert)
            self.assertEqual(len(calls), 1)
            self.assertEqual(samples.shape, (4_000,))
            self.assertFalse((root / "tmp" / "convert-clip.wav").exists())

    def test_unsupported_suffix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "notes.pdf"
            path.write_bytes(b"pdf")
            with self.assertRaisesRegex(AudioFileError, "wav, mp3, m4a"):
                load_mono_16k(path, temp_dir=Path(temp))

    def test_long_audio_is_split_with_overlap_and_joined_text(self):
        samples = np.zeros(int(75 * SAMPLE_RATE), dtype=np.float32)
        chunks = chunk_samples(samples)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].size, 30 * SAMPLE_RATE)
        self.assertEqual(chunks[1].size, 30 * SAMPLE_RATE)
        self.assertEqual(chunks[2].size, 17 * SAMPLE_RATE)
        self.assertEqual(join_transcripts([" hello ", "", "world "]), "hello world")

    def test_short_audio_stays_one_chunk(self):
        samples = np.zeros(int(12 * SAMPLE_RATE), dtype=np.float32)
        self.assertEqual(len(chunk_samples(samples)), 1)

    def test_transcript_save_and_chunk_wav_are_private(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text_path = root / "out.txt"
            write_transcript_text(text_path, "private words")
            wav_path = write_canonical_wav(
                np.zeros(160, dtype=np.float32), root / "chunks"
            )
            self.assertEqual(text_path.read_text(encoding="utf-8"), "private words")
            self.assertEqual(text_path.stat().st_mode & 0o777, 0o600)
            self.assertTrue(wav_path.exists())
            self.assertEqual(wav_path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
