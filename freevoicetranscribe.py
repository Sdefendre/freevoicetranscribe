#!/usr/bin/env python3
"""
FreeVoiceTranscribe — Fast local macOS dictation with lightning-whisper-mlx.

Hold Fn to talk, release to transcribe. Text is typed into the active app.
A floating waveform overlay shows recording/transcribing status.
"""

import os
import sys
import time
import wave
import tempfile
import threading

import numpy as np
import pyaudio
import rumps
from pynput import keyboard as pynput_keyboard

from fn_listener import FnListener
from overlay import Overlay

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024

# Model
DEFAULT_MODEL = "distil-large-v3"


class Recorder:
    """Captures audio from the microphone."""

    def __init__(self):
        self.recording = False
        self._frames = []
        self._stream = None
        self._pa = None
        self._thread = None
        self.on_audio_chunk = None  # callback(np.array int16)

    def start(self):
        self.recording = True
        self._frames = []
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._thread.start()

    def _record(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=FRAMES_PER_BUFFER,
            input=True,
        )

        while self.recording:
            data = self._stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            self._frames.append(data)
            if self.on_audio_chunk:
                chunk = np.frombuffer(data, dtype=np.int16)
                self.on_audio_chunk(chunk)

        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def stop(self):
        self.recording = False
        if self._thread:
            self._thread.join(timeout=2)

    def save_wav(self):
        """Save recorded audio to a temp WAV file. Returns the file path."""
        if not self._frames:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(self._frames))
        return tmp.name


class Transcriber:
    """Transcribes audio using lightning-whisper-mlx."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self._model_name = model_name
        self._whisper = None

    def _ensure_model(self):
        if self._whisper is None:
            from lightning_whisper_mlx import LightningWhisperMLX
            print(f"Loading model: {self._model_name}")
            self._whisper = LightningWhisperMLX(
                model=self._model_name, batch_size=12
            )
            print("Model loaded.")

    def transcribe(self, wav_path):
        """Transcribe a WAV file and return the text."""
        self._ensure_model()
        result = self._whisper.transcribe(wav_path)
        text = result.get("text", "").strip()
        return text


class TextTyper:
    """Types text into the active application using pynput."""

    def __init__(self):
        self._kb = pynput_keyboard.Controller()

    def type_text(self, text):
        for ch in text:
            try:
                self._kb.type(ch)
                time.sleep(0.002)
            except Exception:
                pass


class StatusBarApp(rumps.App):
    """macOS menu bar app with mic icon."""

    def __init__(self, app):
        super().__init__("FreeVoiceTranscribe", "🎙️")
        self._app = app
        self.menu = [
            rumps.MenuItem("Hold Fn to dictate", callback=None),
            None,
        ]

    @rumps.clicked("Quit")
    def quit_app(self, _):
        self._app.shutdown()
        rumps.quit_application()


class App:
    """Main application controller."""

    def __init__(self):
        self.recorder = Recorder()
        self.transcriber = Transcriber()
        self.typer = TextTyper()
        self.overlay = Overlay()
        self.fn_listener = FnListener(
            on_press=self._on_fn_press,
            on_release=self._on_fn_release,
        )
        self._recording = False
        self._processing = False

    def _on_fn_press(self):
        """Called from fn_listener thread when Fn is pressed."""
        if self._recording or self._processing:
            return
        self._recording = True
        self.recorder.on_audio_chunk = self.overlay.update_audio
        self.recorder.start()
        # Show overlay on main thread
        self._perform_on_main(self.overlay.show)
        self._perform_on_main(self._update_menubar_recording, True)

    def _on_fn_release(self):
        """Called from fn_listener thread when Fn is released."""
        if not self._recording:
            return
        self._recording = False
        self._processing = True
        self.recorder.stop()

        # Show transcribing state on main thread
        self._perform_on_main(self.overlay.set_transcribing)

        # Transcribe in background thread
        threading.Thread(target=self._transcribe_and_type, daemon=True).start()

    def _transcribe_and_type(self):
        wav_path = self.recorder.save_wav()
        if wav_path is None:
            self._processing = False
            self._perform_on_main(self.overlay.hide)
            return

        try:
            text = self.transcriber.transcribe(wav_path)
            if text:
                self.typer.type_text(text)
        except Exception as e:
            print(f"Transcription error: {e}")
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
            self._processing = False
            self._perform_on_main(self.overlay.hide)
            self._perform_on_main(self._update_menubar_recording, False)

    def _update_menubar_recording(self, is_recording):
        if hasattr(self, '_statusbar'):
            self._statusbar.title = "🔴" if is_recording else "🎙️"

    def _perform_on_main(self, fn, *args):
        """Schedule a function on the main thread via rumps timer."""
        from PyObjCTools import AppHelper
        AppHelper.callAfter(fn, *args)

    def run(self):
        self.fn_listener.start()
        print("FreeVoiceTranscribe is running.")
        print("Hold the Fn key to dictate. Release to transcribe.")
        self._statusbar = StatusBarApp(self)
        self._statusbar.run()

    def shutdown(self):
        self.fn_listener.stop()
        if self._recording:
            self.recorder.stop()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()
