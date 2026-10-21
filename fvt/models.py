"""Local speech-model catalog used by settings and the transcriber.

FreeVoiceTranscribe stays on lightning-whisper-mlx. Parakeet would need a
different inference stack, so Fast is a smaller Whisper variant instead.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Accurate remains the original first-run default.
DEFAULT_PROFILE_ID = "accurate"
FAST_PROFILE_ID = "fast"
ACCURATE_PROFILE_ID = "accurate"


@dataclass(frozen=True, slots=True)
class SpeechModelProfile:
    """One user-facing model choice and the files lightning-whisper-mlx expects."""

    id: str
    label: str
    model_name: str
    repo_id: str
    summary: str
    detail: str
    # Distil checkpoints live in one Hub repo with paths like
    # ./mlx_models/<name>/weights.npz. Other Whisper models store the two
    # files at the repo root and we place them in mlx_models/<name>/.
    nested_repo_paths: bool


PROFILES: dict[str, SpeechModelProfile] = {
    FAST_PROFILE_ID: SpeechModelProfile(
        id=FAST_PROFILE_ID,
        label="Fast",
        model_name="small",
        repo_id="mlx-community/whisper-small-mlx",
        summary="Whisper small. Quicker on-device transcription.",
        detail="A smaller multilingual model. Slightly less precise than Accurate.",
        nested_repo_paths=False,
    ),
    ACCURATE_PROFILE_ID: SpeechModelProfile(
        id=ACCURATE_PROFILE_ID,
        label="Accurate",
        model_name="distil-large-v3",
        repo_id="mustafaaljadery/distil-whisper-mlx",
        summary="distil-large-v3. Best quality, still the default.",
        detail="The original high-quality local model.",
        nested_repo_paths=True,
    ),
}

PROFILE_ORDER: tuple[str, ...] = (FAST_PROFILE_ID, ACCURATE_PROFILE_ID)


def normalize_profile(value: object) -> str:
    """Return a known profile id, falling back to Accurate."""
    if isinstance(value, str) and value in PROFILES:
        return value
    return DEFAULT_PROFILE_ID


def profile_for(value: object) -> SpeechModelProfile:
    return PROFILES[normalize_profile(value)]


def model_name_for_profile(value: object) -> str:
    return profile_for(value).model_name


def profile_for_model_name(model_name: str) -> SpeechModelProfile | None:
    for profile in PROFILES.values():
        if profile.model_name == model_name:
            return profile
    return None


def remote_filenames(profile: SpeechModelProfile) -> tuple[str, str]:
    """Hub filenames that lightning-whisper-mlx downloads for this model."""
    if profile.nested_repo_paths:
        prefix = f"./mlx_models/{profile.model_name}"
        return (f"{prefix}/weights.npz", f"{prefix}/config.json")
    return ("weights.npz", "config.json")


def local_model_dir(model_root: Path, profile: SpeechModelProfile) -> Path:
    return model_root / profile.model_name


def download_profile(
    profile: SpeechModelProfile,
    *,
    model_root: Path,
    on_progress: Callable[[float], None] | None = None,
    download_file: Callable[..., str] | None = None,
) -> None:
    """Download the two files lightning-whisper-mlx needs, with progress.

    ``download_file`` is the Hub download function. Tests inject a fake so this
    stays local and offline. The default is huggingface_hub.hf_hub_download.
    """
    weights_name, config_name = remote_filenames(profile)
    destination = local_model_dir(model_root, profile)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)

    if profile.nested_repo_paths:
        # Match lightning-whisper-mlx: those filenames already include mlx_models/.
        local_dir = str(model_root.parent)
    else:
        local_dir = str(destination)

    files = (config_name, weights_name)
    completed = 0
    if on_progress is not None:
        on_progress(0.0)

    downloader = download_file
    if downloader is None:
        # Imported only for an on-demand download so catalog tests stay light.
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    for filename in files:
        tqdm_class = None
        if on_progress is not None:
            start = completed / len(files)
            span = 1.0 / len(files)
            tqdm_class = _progress_tqdm_class(on_progress, start=start, span=span)
        kwargs: dict[str, object] = {
            "repo_id": profile.repo_id,
            "filename": filename,
            "local_dir": local_dir,
        }
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class
        try:
            downloader(**kwargs)
        except TypeError:
            # Older Hub clients and test doubles may not accept tqdm_class.
            kwargs.pop("tqdm_class", None)
            downloader(**kwargs)
        completed += 1
        if on_progress is not None:
            on_progress(completed / len(files))


def _progress_tqdm_class(
    on_progress: Callable[[float], None], *, start: float, span: float
):
    """Build a tqdm stand-in that maps one file's bytes onto a global fraction."""

    class _ProgressTqdm:
        def __init__(self, *args, **kwargs):
            self.n = 0
            total = kwargs.get("total")
            if total is None and args:
                try:
                    total = len(args[0])
                except TypeError:
                    total = 0
            self.total = float(total or 0)
            self.disable = False

        def update(self, n=1):
            self.n += n
            if self.total > 0:
                fraction = start + span * min(1.0, self.n / self.total)
                on_progress(min(1.0, fraction))

        def close(self):
            if self.total > 0:
                on_progress(min(1.0, start + span))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            self.close()

        def set_description(self, *args, **kwargs):
            return self

        def set_postfix(self, *args, **kwargs):
            return self

        def refresh(self, *args, **kwargs):
            return None

    return _ProgressTqdm
