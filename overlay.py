"""Backward-compatible imports for the modular native HUD."""

from fvt.ui import HUDView as WaveformView
from fvt.ui import RecordingHUD as Overlay

__all__ = ["Overlay", "WaveformView"]
