# Contributing to FreeVoiceTranscribe

Thanks for your interest. This is a small, opinionated macOS dictation tool.
If you want to change behavior, start with a minimal patch or a bug report.

## Code of Conduct

This project follows the Contributor Covenant. See `CODE_OF_CONDUCT.md`.

## Getting Started

Install Homebrew Python 3.11 and PortAudio, then clone the repository and create the virtual environment:

```bash
brew install python@3.11 portaudio
git clone https://github.com/Sdefendre/freevoicetranscribe.git
cd freevoicetranscribe
./run.sh
```

Install the development tools before running the checks:

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

## Development Rules

- macOS only. Do not add Linux or Windows workarounds.
- Do not hardcode personal paths, accounts, or non-public URLs.
- Do not add network transmission by default.
- Prefer small services in `fvt/` over monolithic logic.
- Preserve existing hotkeys, audio contract, and insertion behavior unless
  the change is explicitly scoped to behavior changes.

## Branching and Commits

Use branch names like:

- `feat/short-description`
- `fix/short-description`
- `docs/short-description`
- `ci/short-description`

Write commit messages in imperative mood, with a short body if needed:

```
fix: restore clipboard when paste is denied
```

## Pull Requests

Keep PRs small. Update `README.md` when the change affects user behavior,
setup, supported versions, or permissions.

CI runs `unittest`, `ruff`, and script syntax checks. If CI fails, fix the
first failure first.

## Releases

Tag builds attach a ZIP and checksum to a draft GitHub Release. The CI app is
ad-hoc signed for review; public distribution requires Developer ID signing,
notarization, stapling, and clean-Mac verification. See the README release steps.
Source contributors should not change the build scripts or bundle metadata
without discussion.

## Questions?

Open a GitHub Discussion or issue.
