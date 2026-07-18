"""macOS privacy permission checks and recovery actions."""

from __future__ import annotations

import logging
from collections.abc import Callable

import AppKit
import ApplicationServices
import objc
from Foundation import NSURL, NSBundle

from .state import PermissionState

_AV_MEDIA_TYPE_AUDIO = "soun"


def _load_capture_device_class():
    try:
        bundle = NSBundle.bundleWithPath_(
            "/System/Library/Frameworks/AVFoundation.framework"
        )
        if bundle is not None:
            bundle.load()
        return objc.lookUpClass("AVCaptureDevice")
    except Exception:
        return None


class MacPermissionService:
    def __init__(
        self,
        *,
        capture_device=None,
        ax_trusted: Callable[[], bool] | None = None,
        workspace=None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._capture_device = capture_device or _load_capture_device_class()
        self._ax_trusted = ax_trusted or ApplicationServices.AXIsProcessTrusted
        self._workspace = workspace or AppKit.NSWorkspace.sharedWorkspace()
        self._logger = logger or logging.getLogger("freevoicetranscribe.permissions")

    def accessibility_state(self) -> PermissionState:
        try:
            return (
                PermissionState.GRANTED
                if bool(self._ax_trusted())
                else PermissionState.DENIED
            )
        except Exception:
            return PermissionState.UNKNOWN

    def microphone_state(self) -> PermissionState:
        device = self._capture_device
        if device is None:
            return PermissionState.UNKNOWN
        try:
            status = int(device.authorizationStatusForMediaType_(_AV_MEDIA_TYPE_AUDIO))
        except Exception:
            self._logger.debug("Could not read microphone permission", exc_info=True)
            return PermissionState.UNKNOWN
        return {
            0: PermissionState.UNKNOWN,
            1: PermissionState.RESTRICTED,
            2: PermissionState.DENIED,
            3: PermissionState.GRANTED,
        }.get(status, PermissionState.UNKNOWN)

    def request_microphone(self, callback: Callable[[bool], None]) -> None:
        device = self._capture_device
        if device is None:
            self.open_microphone_settings()
            callback(False)
            return

        state = self.microphone_state()
        if state is PermissionState.GRANTED:
            callback(True)
            return
        if state in {PermissionState.DENIED, PermissionState.RESTRICTED}:
            self.open_microphone_settings()
            callback(False)
            return

        try:
            device.requestAccessForMediaType_completionHandler_(
                _AV_MEDIA_TYPE_AUDIO,
                lambda granted: callback(bool(granted)),
            )
        except Exception:
            self._logger.debug("Microphone permission request failed", exc_info=True)
            self.open_microphone_settings()
            callback(False)

    def open_accessibility_settings(self) -> None:
        self._open_settings("Privacy_Accessibility")

    def open_microphone_settings(self) -> None:
        self._open_settings("Privacy_Microphone")

    def _open_settings(self, pane: str) -> None:
        url = NSURL.URLWithString_(
            f"x-apple.systempreferences:com.apple.preference.security?{pane}"
        )
        if url is not None:
            self._workspace.openURL_(url)
