# Security Policy

## Supported Versions

FreeVoiceTranscribe is early-stage software. Updates and security fixes are
provided for the current main branch. Use the latest release when possible.

## Reporting a Vulnerability

Report security issues privately rather than opening a public GitHub issue.

- Email: steve@defendresolutions.com
- Subject: `SECURITY: FreeVoiceTranscribe`

Please include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- macOS and Python versions
- Whether the issue involves source runs, a built app, or both

The reporter will be acknowledged in release notes unless anonymity is requested.

## Scope

- Source code in this repository
- Built artifacts published by FreeVoiceTranscribe
- GitHub Actions workflows and release process

Out of scope:

- Third-party dependencies such as `lightning-whisper-mlx`, `mlx`, `rumps`,
  `pynput`, or `PyAudio`
- macOS itself or the Apple notarization/signing pipeline

## Safe Usage Notes

- FreeVoiceTranscribe opens the microphone only while recording.
- It keeps the latest transcript in `~/Library/Application Support/FreeVoiceTranscribe/`.
- It does not transmit audio or transcripts over the network by default.
- If you add telemetry or crash reporting, make it opt-in and document it here.
