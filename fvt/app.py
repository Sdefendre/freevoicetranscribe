"""Application composition root."""

from __future__ import annotations

import logging
import os

from PyObjCTools import AppHelper

from .audio import AudioRecorder
from .coordinator import AppCoordinator
from .hotkey import FnListener
from .insertion import MacTextInserter
from .permissions import MacPermissionService
from .storage import (
    AppPaths,
    DeviceSettingsStore,
    LastTranscriptStore,
    configure_logging,
)
from .transcription import MLXTranscriber
from .ui import StatusBarApp


def create_application(paths: AppPaths | None = None):
    paths = paths or AppPaths.default()
    paths.configure_runtime_environment()
    logger = configure_logging(paths)
    removed = paths.remove_stale_audio()
    if removed:
        logger.info("Removed %d stale temporary audio file(s)", removed)

    coordinator_ref: dict[str, AppCoordinator] = {}
    hotkey = FnListener(
        on_press=lambda: coordinator_ref["coordinator"].fn_press(),
        on_release=lambda: coordinator_ref["coordinator"].fn_release(),
        on_fn_space=lambda: coordinator_ref["coordinator"].fn_space(),
        on_cancel=lambda: coordinator_ref["coordinator"].cancel_recording(),
        logger=logger.getChild("hotkey"),
    )
    settings = DeviceSettingsStore(paths.device_settings_path())
    coordinator = AppCoordinator(
        audio=AudioRecorder(
            paths.temp,
            logger=logger.getChild("audio"),
            input_device_index=settings.load().get("input_device_index"),
        ),
        transcriber=MLXTranscriber(paths, logger=logger.getChild("transcription")),
        inserter=MacTextInserter(logger=logger.getChild("insertion")),
        hotkey=hotkey,
        permissions=MacPermissionService(logger=logger.getChild("permissions")),
        logger=logger.getChild("coordinator"),
        settings_store=settings,
        transcript_store=LastTranscriptStore.default(),
    )
    coordinator_ref["coordinator"] = coordinator
    status_bar = StatusBarApp(coordinator)
    coordinator.set_observers(
        on_snapshot=lambda snapshot: AppHelper.callAfter(status_bar.update, snapshot),
        on_audio_level=status_bar.hud.update_audio,
    )
    return coordinator, status_bar


def main() -> None:
    coordinator = None
    try:
        coordinator, status_bar = create_application()
        coordinator.start()
        if os.environ.get("FVT_LAUNCH_SMOKE") == "1":
            print("FVT_APP_READY", flush=True)
        status_bar.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.getLogger("freevoicetranscribe").exception("Application startup failed")
        raise
    finally:
        if coordinator is not None:
            coordinator.shutdown()


if __name__ == "__main__":
    main()
