import tempfile
import threading
import time
import unittest
from pathlib import Path

from fvt.audio import AudioError, AudioRecorder


class FakeStream:
    def __init__(self):
        self.closed = False

    def read(self, frame_count, exception_on_overflow=False):
        time.sleep(0.001)
        return b"\x01\x00" * frame_count

    def stop_stream(self):
        pass

    def abort_stream(self):
        pass

    def close(self):
        self.closed = True


class FakePyAudio:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.stream = FakeStream()
        self.terminated = False

    def open(self, **_kwargs):
        if self.fail:
            raise OSError("microphone denied")
        return self.stream

    def terminate(self):
        self.terminated = True


class BlockingStream(FakeStream):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.abort_calls = 0

    def read(self, frame_count, exception_on_overflow=False):
        self.release.wait()
        return b"\x01\x00" * frame_count

    def abort_stream(self):
        self.abort_calls += 1


class AudioRecorderTests(unittest.TestCase):
    def test_start_handshake_and_bounded_stop_create_private_wav(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = AudioRecorder(
                Path(temp),
                max_seconds=1,
                pa_factory=FakePyAudio,
            )
            recorder.start()
            time.sleep(0.01)
            captured = recorder.stop()
            self.assertTrue(captured.path.exists())
            self.assertGreater(captured.duration_seconds, 0)
            self.assertEqual(captured.path.stat().st_mode & 0o777, 0o600)
            captured.cleanup()
            self.assertFalse(captured.path.exists())

    def test_microphone_open_failure_is_reported_before_start_returns(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = AudioRecorder(
                Path(temp),
                pa_factory=lambda: FakePyAudio(fail=True),
            )
            with self.assertRaisesRegex(AudioError, "Could not open"):
                recorder.start()

    def test_cancel_discards_frames_without_writing_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            recorder = AudioRecorder(temp_path, pa_factory=FakePyAudio)
            recorder.start()
            recorder.cancel()
            self.assertEqual(list(temp_path.glob("*.wav")), [])

    def test_stop_timeout_retains_session_and_blocks_second_start(self):
        with tempfile.TemporaryDirectory() as temp:
            audio_api = FakePyAudio()
            audio_api.stream = BlockingStream()
            recorder = AudioRecorder(
                Path(temp),
                pa_factory=lambda: audio_api,
                stop_timeout=0.01,
                abort_timeout=0.01,
            )
            recorder.start()
            with self.assertRaisesRegex(AudioError, "still stopping"):
                recorder.stop()
            self.assertTrue(recorder.is_stopping)
            self.assertGreater(audio_api.stream.abort_calls, 0)
            with self.assertRaisesRegex(AudioError, "previous recording"):
                recorder.start()

            audio_api.stream.release.set()
            self.assertTrue(recorder.cancel())
            self.assertFalse(recorder.is_stopping)


if __name__ == "__main__":
    unittest.main()
