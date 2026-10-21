"""Optional on-device writing model for cleaned-up dictation.

The speech model still produces the raw transcript. This step is off unless
the user chooses Write or Command, and it never sends text off the Mac.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

from .contracts import StatusCallback
from .storage import AppPaths

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:  # pragma: no cover - present after the runtime install
    hf_snapshot_download = None

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
except ImportError:  # pragma: no cover - present after the runtime install
    mlx_generate = None
    mlx_load = None

try:
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler
except ImportError:  # pragma: no cover - older mlx-lm builds omit this helper
    mlx_make_sampler = None

# Apache-2.0 Qwen instruct model, 4-bit MLX build, about 1 GB on disk.
DEFAULT_WRITING_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_WRITING_MODEL_DIR = "Qwen2.5-1.5B-Instruct-4bit"
MAX_NEW_TOKENS = 1024

WRITE_SYSTEM_PROMPT = """You edit dictated speech into written text.
Remove filler words such as um, uh, like, you know, and I mean.
Fix punctuation and capitalization.
Apply spoken self-corrections. Example: "the red no the blue folder" becomes "the blue folder".
Do not add facts, greetings, or explanations.
Reply with only the edited text."""

COMMAND_SYSTEM_PROMPT = """You rewrite the selected text using the spoken instruction.
Follow the instruction.
Keep the original meaning unless the instruction changes it.
Do not add a preface, quotes, or explanation.
Reply with only the rewritten text."""


class WritingModelUnavailable(RuntimeError):
    pass


class WritingModelError(RuntimeError):
    pass


LoadFn = Callable[..., tuple[object, object]]
GenerateFn = Callable[..., str]
DownloadFn = Callable[[str, Path, StatusCallback | None], None]


def build_messages(
    text: str,
    *,
    mode: str,
    tone: str = "",
    selected_text: str = "",
) -> list[dict[str, str]]:
    """Build the local chat messages for Write or Command cleanup."""
    if mode == "command":
        system = COMMAND_SYSTEM_PROMPT
        user = (
            f"Selected text:\n{selected_text.strip()}\n\nInstruction:\n{text.strip()}"
        )
    else:
        system = WRITE_SYSTEM_PROMPT
        user = text.strip()
    if tone.strip():
        system = f"{system}\nWriting style: {tone.strip()}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def extract_cleaned_text(raw: str) -> str:
    """Strip common model wrappers without changing the words."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    return text


def is_plausible_cleanup(original: str, cleaned: str, *, mode: str) -> bool:
    """Reject empty or obviously rambling Write-mode expansions."""
    if not cleaned.strip():
        return False
    if mode == "command":
        return True
    return len(cleaned) <= max(400, len(original) * 4)


def _default_download(
    repo_id: str, destination: Path, on_status: StatusCallback | None
) -> None:
    if hf_snapshot_download is None:
        raise WritingModelUnavailable("huggingface_hub is not available")
    if on_status is not None:
        on_status("Downloading writing model")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    hf_snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
    )


def _default_load(model_dir: str) -> tuple[object, object]:
    if mlx_load is None:
        raise WritingModelUnavailable("mlx_lm is not available")
    return mlx_load(model_dir)


def _default_generate(model, tokenizer, prompt: str) -> str:
    if mlx_generate is None:
        raise WritingModelUnavailable("mlx_lm is not available")
    if mlx_make_sampler is not None:
        try:
            return mlx_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_NEW_TOKENS,
                sampler=mlx_make_sampler(temp=0.1),
                verbose=False,
            )
        except TypeError:
            pass
    return mlx_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        verbose=False,
    )


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return "\n\n".join(item["content"] for item in messages) + "\n"
    try:
        rendered = apply(messages, tokenize=False, add_generation_prompt=True)
    except TypeError:
        rendered = apply(messages, add_generation_prompt=True)
    if isinstance(rendered, str):
        return rendered
    decode = getattr(tokenizer, "decode", None)
    if decode is not None:
        return str(decode(rendered))
    return str(rendered)


class LocalTextCleaner:
    """Own the optional local writing model and serialize inference calls."""

    def __init__(
        self,
        paths: AppPaths,
        *,
        model_id: str = DEFAULT_WRITING_MODEL,
        model_dir_name: str = DEFAULT_WRITING_MODEL_DIR,
        load_fn: LoadFn | None = None,
        generate_fn: GenerateFn | None = None,
        download_fn: DownloadFn | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._paths = paths
        self._model_id = model_id
        self._model_dir_name = model_dir_name
        self._load_fn = load_fn
        self._generate_fn = generate_fn
        self._download_fn = download_fn
        self._logger = logger or logging.getLogger("freevoicetranscribe.cleanup")
        self._model = None
        self._tokenizer = None
        self._prepare_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_dir(self) -> Path:
        return self._paths.writing_model_root / self._model_dir_name

    @property
    def model_present(self) -> bool:
        config = self.model_dir / "config.json"
        tokenizer = self.model_dir / "tokenizer.json"
        try:
            if config.stat().st_size <= 0 or tokenizer.stat().st_size <= 0:
                return False
        except OSError:
            return False
        return any(self.model_dir.glob("*.safetensors")) or any(
            self.model_dir.glob("*.npz")
        )

    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def prepare(
        self, *, allow_download: bool, on_status: StatusCallback | None = None
    ) -> None:
        if self.is_ready:
            return
        if not allow_download and not self.model_present:
            raise WritingModelUnavailable(
                "The local writing model has not been downloaded"
            )

        with self._prepare_lock:
            if self.is_ready:
                return
            try:
                if not self.model_present:
                    if not allow_download:
                        raise WritingModelUnavailable(
                            "The local writing model has not been downloaded"
                        )
                    downloader = self._download_fn or _default_download
                    downloader(self._model_id, self.model_dir, on_status)
                if on_status is not None:
                    on_status("Loading writing model")
                loader = self._load_fn or _default_load
                model, tokenizer = loader(str(self.model_dir))
                self._model = model
                self._tokenizer = tokenizer
            except WritingModelUnavailable:
                raise
            except Exception as exc:
                self._logger.warning(
                    "Writing model preparation failed: %s", type(exc).__name__
                )
                raise WritingModelUnavailable(
                    "The writing model could not be prepared. Check the network and free disk space."
                ) from exc
            if on_status is not None:
                on_status("Writing model ready")

    def clean(
        self,
        text: str,
        *,
        mode: str,
        tone: str = "",
        selected_text: str = "",
    ) -> str:
        original = str(text).strip()
        if not original:
            return original
        if self._model is None or self._tokenizer is None:
            raise WritingModelUnavailable("The writing model is not ready")
        messages = build_messages(
            original, mode=mode, tone=tone, selected_text=selected_text
        )
        try:
            with self._inference_lock:
                prompt = render_prompt(self._tokenizer, messages)
                generate = self._generate_fn or _default_generate
                raw = generate(self._model, self._tokenizer, prompt)
            cleaned = extract_cleaned_text(raw)
            if not is_plausible_cleanup(original, cleaned, mode=mode):
                raise WritingModelError("The writing model returned unusable text")
            return cleaned
        except (WritingModelUnavailable, WritingModelError):
            raise
        except Exception as exc:
            self._logger.warning("Local writing cleanup failed: %s", type(exc).__name__)
            raise WritingModelError("Local writing cleanup failed. Try again.") from exc
