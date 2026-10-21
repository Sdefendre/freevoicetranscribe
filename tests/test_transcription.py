import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from fvt.models import profile_for
from fvt.storage import AppPaths
from fvt.transcription import MLXTranscriber, ModelUnavailable


class FakeModel:
    def __init__(self, **_kwargs):
        pass

    def transcribe(self, _path):
        return {"text": "  local transcript  "}


class TranscriptionTests(unittest.TestCase):
    @staticmethod
    def _write_wav(path):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(b"\x00\x00" * 160)

    def test_prepare_requires_local_model_without_download_permission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "tmp"
            )
            service = MLXTranscriber(paths, model_factory=FakeModel)
            with self.assertRaises(ModelUnavailable):
                service.prepare(allow_download=False)

    def test_model_contract_returns_trimmed_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "tmp"
            )
            model_dir = paths.model_root / "distil-large-v3"
            model_dir.mkdir(parents=True)
            (model_dir / "weights.npz").write_bytes(b"model")
            (model_dir / "config.json").write_text("{}")
            factory_calls = []

            def network_capable_factory(**kwargs):
                factory_calls.append(kwargs)
                return FakeModel(**kwargs)

            service = MLXTranscriber(
                paths,
                model_factory=network_capable_factory,
                transcribe_callable=lambda *_args, **_kwargs: {
                    "text": "  local transcript  "
                },
            )
            service.prepare(allow_download=False)
            self.assertTrue(service.is_ready)
            self.assertEqual(factory_calls, [])
            audio_path = root / "audio.wav"
            self._write_wav(audio_path)
            self.assertEqual(service.transcribe(audio_path), "local transcript")
            self.assertEqual(
                service.transcribe_samples(np.zeros(160, dtype=np.float32)),
                "local transcript",
            )

    def test_invalid_local_config_fails_without_calling_factory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "tmp"
            )
            model_dir = paths.model_root / "distil-large-v3"
            model_dir.mkdir(parents=True)
            (model_dir / "weights.npz").write_bytes(b"model")
            (model_dir / "config.json").write_text("not json")
            calls = []
            service = MLXTranscriber(
                paths, model_factory=lambda **kwargs: calls.append(kwargs)
            )
            with self.assertRaises(ModelUnavailable):
                service.prepare(allow_download=False)
            self.assertEqual(calls, [])

    def test_fast_model_uses_its_own_directory_and_can_switch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "tmp"
            )
            fast_dir = paths.model_root / "small"
            fast_dir.mkdir(parents=True)
            (fast_dir / "weights.npz").write_bytes(b"fast")
            (fast_dir / "config.json").write_text("{}")
            downloads = []

            def fake_download(profile, **kwargs):
                downloads.append((profile.id, kwargs["model_root"]))

            service = MLXTranscriber(
                paths,
                model_name="distil-large-v3",
                download_callable=fake_download,
            )
            self.assertEqual(service.model_dir.name, "distil-large-v3")
            service.set_model_name("small")
            self.assertFalse(service.is_ready)
            self.assertTrue(service.model_present)
            service.prepare(allow_download=False)
            self.assertTrue(service.is_ready)
            self.assertEqual(downloads, [])

    def test_download_reports_progress_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = AppPaths(
                root / "support", root / "cache", root / "logs", root / "tmp"
            )
            fractions = []
            statuses = []

            def fake_download(profile, **kwargs):
                on_progress = kwargs["on_progress"]
                dest = paths.model_root / profile.model_name
                dest.mkdir(parents=True)
                (dest / "weights.npz").write_bytes(b"model")
                (dest / "config.json").write_text("{}")
                if on_progress is not None:
                    on_progress(0.4)
                    on_progress(1.0)
                fractions.append(profile.id)

            service = MLXTranscriber(
                paths,
                model_name=profile_for("fast").model_name,
                download_callable=fake_download,
            )
            service.prepare(
                allow_download=True,
                on_status=statuses.append,
                on_progress=fractions.append,
            )
            self.assertIn("fast", fractions)
            self.assertIn(0.4, fractions)
            self.assertIn(1.0, fractions)
            self.assertIn("Downloading Fast model", statuses)
            self.assertTrue(service.is_ready)


if __name__ == "__main__":
    unittest.main()
