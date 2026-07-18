"""py2app configuration for the distributable macOS application."""

import sys
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
ICON = ROOT / ".build-assets" / "AppIcon.icns"
sys.setrecursionlimit(10_000)

BUILDING_APP = "py2app" in sys.argv
APP = [str(ROOT / "scripts" / "app_entry.py")]
OPTIONS = {
    "argv_emulation": False,
    "extra_scripts": [str(ROOT / "scripts" / "bundle_import_probe.py")],
    "iconfile": str(ICON),
    "packages": ["fvt", "lightning_whisper_mlx"],
    "includes": [
        "AppKit",
        "ApplicationServices",
        "Quartz",
        "mlx",
        # Imported dynamically by the MLX native extension during PyInit_core.
        # modulegraph cannot discover this dependency from Python source.
        "mlx._reprlib_fix",
        "numba",
        "numpy",
        "pyaudio",
        "pynput",
        # pynput selects both backends with importlib at package import time.
        "pynput._util.darwin",
        "pynput.keyboard._darwin",
        "pynput.mouse._darwin",
        "rumps",
    ],
    "excludes": [
        "IPython",
        "matplotlib",
        "pandas",
        "pytest",
        "setuptools",
        "torch",
        "lightning_whisper_mlx.torch_whisper",
        "wheel",
    ],
    "plist": {
        "CFBundleName": "FreeVoiceTranscribe",
        "CFBundleDisplayName": "FreeVoiceTranscribe",
        "CFBundleIdentifier": "com.stevedefendre.freevoicetranscribe",
        "CFBundleShortVersionString": "2.0.0",
        "CFBundleVersion": "2.0.0",
        "LSMinimumSystemVersion": "14.0",
        "LSUIElement": True,
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": (
            "FreeVoiceTranscribe records only while you dictate and transcribes "
            "the audio locally on this Mac."
        ),
    },
}

setup_options = {}
if BUILDING_APP:
    from py2app.build_app import py2app as Py2AppCommand

    class LockedPy2App(Py2AppCommand):
        """Dependencies are preinstalled from the release lock by build_app.sh."""

        def finalize_options(self):
            # py2app 0.28 rejects install_requires. Keep correct wheel metadata in
            # pyproject.toml while clearing it only for the application command.
            self.distribution.install_requires = []
            super().finalize_options()

    setup_options = {
        "app": APP,
        "options": {"py2app": OPTIONS},
        "cmdclass": {"py2app": LockedPy2App},
    }

setup(**setup_options)
