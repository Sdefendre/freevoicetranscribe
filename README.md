# FreeVoiceTranscribe

Fast local macOS dictation using [lightning-whisper-mlx](https://github.com/mustafaaljadery/lightning-whisper-mlx) on Apple Silicon.

Hold the **Fn key** to talk, release to transcribe. Text is typed directly into the active app. A floating waveform overlay shows recording status.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- PortAudio (`brew install portaudio`)

## Setup

```bash
chmod +x run.sh
./run.sh
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python freevoicetranscribe.py
```

## Permissions

On first run, macOS will prompt for:

- **Microphone** — required for audio capture
- **Accessibility** — required for Fn key detection and text typing

Grant both in System Settings → Privacy & Security.

## Usage

1. Run the app — a mic icon appears in the menu bar
2. Hold **Fn** → overlay appears, start speaking
3. Release **Fn** → audio is transcribed and typed into the active text field

## Model

Uses `distil-large-v3` by default (best speed/quality balance). The model downloads automatically on first run.
