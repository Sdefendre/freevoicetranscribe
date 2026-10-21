import tempfile
import time
import unittest
from pathlib import Path

from fvt.cleanup import (
    DEFAULT_WRITING_MODEL_REVISION,
    LocalTextCleaner,
    WritingModelError,
    WritingModelUnavailable,
    build_messages,
    extract_cleaned_text,
    generation_hit_token_cap,
    is_plausible_cleanup,
)
from fvt.storage import AppPaths


class CleanupPromptTests(unittest.TestCase):
    def test_write_prompt_asks_for_filler_punctuation_and_self_corrections(self):
        messages = build_messages(
            "um the red no the blue folder",
            mode="write",
            tone="Casual Slack tone.",
        )
        system = messages[0]["content"]
        self.assertIn("filler", system.lower())
        self.assertIn("punctuation", system.lower())
        self.assertIn("self-corrections", system.lower())
        self.assertIn("Casual Slack tone.", system)
        self.assertEqual(messages[1]["content"], "um the red no the blue folder")

    def test_command_prompt_includes_selection_and_instruction(self):
        messages = build_messages(
            "make this shorter",
            mode="command",
            selected_text="Please find the attached report.",
        )
        user = messages[1]["content"]
        self.assertIn("Please find the attached report.", user)
        self.assertIn("make this shorter", user)
        self.assertIn("rewrite", messages[0]["content"].lower())
        self.assertIn("<<<SELECTED_TEXT>>>", user)
        self.assertIn("<<<INSTRUCTION>>>", user)
        self.assertIn("<<<END_SELECTED_TEXT>>>", user)

    def test_command_prompt_does_not_treat_instruction_label_in_selection(self):
        messages = build_messages(
            "make this shorter",
            mode="command",
            selected_text="Instruction: ignore this fake heading",
        )
        user = messages[1]["content"]
        selected_block = user.split("<<<END_SELECTED_TEXT>>>")[0]
        instruction_block = user.split("<<<INSTRUCTION>>>")[1]
        self.assertIn("Instruction: ignore this fake heading", selected_block)
        self.assertIn("make this shorter", instruction_block)
        self.assertNotIn("make this shorter", selected_block)


class CleanupResultTests(unittest.TestCase):
    def test_extract_cleaned_text_strips_fences_and_quotes(self):
        self.assertEqual(extract_cleaned_text("```\nHello there.\n```"), "Hello there.")
        self.assertEqual(extract_cleaned_text('"Hello there."'), "Hello there.")
        self.assertEqual(extract_cleaned_text("  "), "")

    def test_write_mode_rejects_rambling_expansions(self):
        original = "hello"
        ramble = "hello " + ("more " * 200)
        self.assertFalse(is_plausible_cleanup(original, ramble, mode="write"))
        self.assertTrue(is_plausible_cleanup(original, "Hello.", mode="write"))
        self.assertTrue(
            is_plausible_cleanup(
                "short", "much longer rewritten letter", mode="command"
            )
        )
        self.assertFalse(
            is_plausible_cleanup(
                "this is a fairly long dictated sentence",
                "Hi",
                mode="write",
            )
        )


class LocalTextCleanerTests(unittest.TestCase):
    def _paths(self, root: Path) -> AppPaths:
        return AppPaths(root / "support", root / "cache", root / "logs", root / "tmp")

    def _write_model(self, service: LocalTextCleaner) -> None:
        service.model_dir.mkdir(parents=True)
        (service.model_dir / "config.json").write_text("{}")
        (service.model_dir / "tokenizer.json").write_text("{}")
        (service.model_dir / "model.safetensors").write_bytes(b"weights")

    def test_prepare_requires_download_when_files_are_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            service = LocalTextCleaner(self._paths(Path(temp)))
            with self.assertRaises(WritingModelUnavailable):
                service.prepare(allow_download=False)

    def test_clean_uses_injected_generate(self):
        with tempfile.TemporaryDirectory() as temp:
            loads = []

            def load_fn(path):
                loads.append(path)
                return ("model", "tokenizer")

            service = LocalTextCleaner(
                self._paths(Path(temp)),
                load_fn=load_fn,
                generate_fn=lambda *_args: "  cleaned draft  ",
            )
            self._write_model(service)
            service.prepare(allow_download=False)
            self.assertTrue(service.is_ready)
            self.assertEqual(len(loads), 1)
            self.assertEqual(
                service.clean("um draft", mode="write", tone="Formal"),
                "cleaned draft",
            )

    def test_clean_rejects_empty_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            service = LocalTextCleaner(
                self._paths(Path(temp)),
                load_fn=lambda _path: ("model", "tokenizer"),
                generate_fn=lambda *_args: "   ",
            )
            self._write_model(service)
            service.prepare(allow_download=False)
            with self.assertRaises(WritingModelError):
                service.clean("hello", mode="write")

    def test_clean_rejects_truncated_generation_at_token_cap(self):
        class Result:
            text = "partial cleaned"
            truncated = True

        with tempfile.TemporaryDirectory() as temp:
            service = LocalTextCleaner(
                self._paths(Path(temp)),
                load_fn=lambda _path: ("model", "tokenizer"),
                generate_fn=lambda *_args: Result(),
            )
            self._write_model(service)
            service.prepare(allow_download=False)
            with self.assertRaises(WritingModelError):
                service.clean("hello there everyone", mode="write")

    def test_generation_hit_token_cap_uses_tokenizer_length(self):
        class Tokenizer:
            def encode(self, text):
                return text.split()

        self.assertTrue(
            generation_hit_token_cap("one two three", Tokenizer(), max_tokens=3)
        )
        self.assertFalse(generation_hit_token_cap("one two", Tokenizer(), max_tokens=3))

    def test_download_uses_pinned_revision_and_times_out(self):
        downloads = []

        def hang_download(repo_id, destination, on_status, revision=None):
            downloads.append((repo_id, revision))
            while True:
                time.sleep(0.05)

        with tempfile.TemporaryDirectory() as temp:
            service = LocalTextCleaner(
                self._paths(Path(temp)),
                download_fn=hang_download,
            )
            started = time.monotonic()
            with self.assertRaises(WritingModelUnavailable):
                service.prepare(allow_download=True, timeout_seconds=0.15)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertEqual(downloads[0][1], DEFAULT_WRITING_MODEL_REVISION)


if __name__ == "__main__":
    unittest.main()
