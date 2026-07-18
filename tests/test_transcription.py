import tempfile
import unittest
import wave
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
