# FreeVoiceTranscribe

Private, local hold-to-talk dictation for Apple Silicon Macs. Focus an editable field, hold **fn/globe**, speak, and release to insert the transcript where you started.

FreeVoiceTranscribe is free, open-source software released under the [MIT license](./LICENSE).

## What it does

- Hold **fn** to record; release it to transcribe and insert.
- Press **fn + Space** while recording to switch to hands-free mode; press **fn** again to finish.
- Press **Esc** while recording to cancel and delete that recording.
- Recover the latest transcript with **Copy Last Transcript** or **Insert Last Transcript** from the menu bar.
- Transcribe locally with the `distil-large-v3` model through `lightning-whisper-mlx`.
- Limit each recording to five minutes.

FreeVoiceTranscribe has no account system, cloud transcription, transcript history, or always-on microphone. It is a menu-bar app after setup and does not remain in the Dock.

## Requirements

| Use | Requirements |
| --- | --- |
| Run from source | Apple Silicon Mac, macOS 14+, Python 3.11–3.13, PortAudio |
| Build the app | The same hardware and OS, plus a framework-capable arm64 Python 3.11 |
| First setup | Internet access and enough free space for the one-time speech-model download |

Install PortAudio with Homebrew:

```bash
brew install portaudio
```

Homebrew Python 3.11 is the recommended release-build interpreter. Static or standalone Python distributions are rejected because `py2app` cannot build a complete runtime from them.

## Run from source

The launcher creates `.venv` with the first compatible Python it finds, installs the pinned runtime requirements when `requirements.txt` changes, and starts the package entry point:

```bash
./run.sh
```

You can also create the environment manually:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH="$PWD" .venv/bin/python -m fvt
```

For a regular package install, the same direct dependencies are declared in `pyproject.toml`:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/freevoicetranscribe
```

`requirements.txt` pins direct source dependencies. `requirements.lock` contains the fully resolved runtime environment, while `requirements-build.lock` also includes build tooling. To reproduce the locked runtime:

```bash
.venv/bin/python -m pip install -c requirements.lock .
```

## First launch

The setup window guides you through three prerequisites:

1. **Microphone access** — captures only recordings you explicitly start.
2. **Accessibility access** — detects the global shortcut, verifies the focused editable field, and pastes the result.
3. **Speech model** — downloads the local `distil-large-v3` model once.

After changing a permission in **System Settings → Privacy & Security**, return to FreeVoiceTranscribe and refresh setup. Packaged and source runs have different macOS permission identities, so switching between them may require granting access again.

Once setup is complete, focus an enabled editable field in another app before pressing **fn**. Apps with custom editors or restricted paste behavior may not expose a compatible field through macOS Accessibility.

## Privacy and local data

- The microphone opens only while recording.
- Inference runs locally after the one-time model download.
- Recordings are private 16 kHz mono temporary WAV files. They are deleted after transcription, error, or cancellation; crash leftovers are removed at the next launch.
- Exactly one transcript is persisted locally for recovery. A newer transcript replaces it; there is no history.
- Automatic insertion temporarily stages text on the clipboard. The previous clipboard is restored only if no other app changed it, so newer clipboard contents are preserved.
- The app captures the original process and exact focused editable Accessibility element. It verifies both again before insertion and fails closed rather than typing into another field.
- If verification or insertion fails, the transcript is copied to the clipboard. If copying also fails, it remains available from the menu bar.
- Rotating diagnostics contain event and error types, never transcript text.

Local files live outside the source tree and app bundle:

```text
~/Library/Application Support/FreeVoiceTranscribe/
├── last-transcript.json       latest transcript only (mode 0600)
└── mlx_models/                downloaded speech model

~/Library/Caches/FreeVoiceTranscribe/
├── Temporary/                 temporary recordings
└── huggingface/               model-download cache

~/Library/Logs/FreeVoiceTranscribe/
├── app.log                    current diagnostic log
├── app.log.1                  rotated log, when present
└── app.log.2                  rotated log, when present
```

Application directories are created with private permissions when the filesystem permits it. `app.log` rotates at 512 KiB with two backups.

## Build the macOS app

The release script creates or refreshes `.build-venv` from `requirements-build.lock`, builds in a staging directory with `py2app`, signs the staged app, verifies it, and publishes only a passing artifact:

```bash
./build_app.sh
```

Successful output is written to:

```text
dist/FreeVoiceTranscribe.app
```

Use a specific compatible interpreter when needed:

```bash
PYTHON=/opt/homebrew/bin/python3.11 ./build_app.sh
```

The default is an ad-hoc signature for local testing. To exercise the hardened-runtime signing path with an installed Developer ID identity:

```bash
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" ./build_app.sh
```

To retain failed staging output for diagnosis instead of deleting it:

```bash
FVT_RETAIN_FAILED_STAGING=1 ./build_app.sh
```

The retained files appear under `build/failed-release/`, never `dist/`.

### Build gates

Before publishing to `dist/`, the build verifies:

- a bundled Python runtime with no copied virtual environment or external symlink;
- required Python and native modules, including MLX and Numba/OpenMP;
- arm64 native code with resolvable library dependencies;
- absence of model files from the signed bundle;
- nested and outer code signatures;
- a terminating isolated import smoke test; and
- normal app startup followed by menu-bar-loop liveness.

The smoke tests do not request Microphone or Accessibility access.

Developer ID signing alone is not a public release. Distribution still requires appropriate entitlements and hardened-runtime review, Apple notarization, stapling, Gatekeeper validation, and testing on a clean supported Mac.

## Development

Install the development tools into the source environment:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

Run the checks:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
bash -n run.sh build_app.sh
```

Validate a candidate release-build interpreter separately:

```bash
/opt/homebrew/bin/python3.11 scripts/validate_build_python.py
```

Verify an existing bundle without rebuilding it:

```bash
.build-venv/bin/python scripts/verify_bundle.py \
  --launch-import-smoke \
  --launch-liveness-smoke \
  dist/FreeVoiceTranscribe.app
```

Use `--skip-signature` only for isolated verifier fixtures; it is not a release validation mode.

## Architecture

```text
fvt/app.py             application composition and lifecycle
fvt/state.py           legal states and immutable snapshots
fvt/coordinator.py     serialized events, workers, and recovery paths
fvt/audio.py           bounded capture and private temporary WAV handling
fvt/transcription.py   local model readiness and PCM transcription
fvt/hotkey.py          resilient fn, fn+Space, and Esc event tap
fvt/insertion.py       exact target validation and safe clipboard insertion
fvt/permissions.py     macOS permission checks and Settings recovery
fvt/storage.py         private paths, latest transcript, and rotating logs
fvt/ui.py              setup window, menu bar, and multi-display HUD
scripts/               build interpreter, bundle, and launch verification
site/                  static noindex product preview
```

The root `freevoicetranscribe.py`, `fn_listener.py`, and `overlay.py` files are compatibility shims for earlier imports and launch commands. New code should use the `fvt` package.

## Contributing

See `CONTRIBUTING.md` for build instructions, testing expectations, and pull-request guidance.

## License and distribution

This project is released under the MIT License. See `LICENSE` for the full terms.
