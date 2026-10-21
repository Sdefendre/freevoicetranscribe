import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fvt.models import (
    ACCURATE_PROFILE_ID,
    DEFAULT_PROFILE_ID,
    FAST_PROFILE_ID,
    PROFILE_ORDER,
    download_profile,
    model_name_for_profile,
    normalize_profile,
    profile_for,
    remote_filenames,
)


class SpeechModelCatalogTests(unittest.TestCase):
    def test_accurate_is_the_default_distil_large_model(self):
        self.assertEqual(DEFAULT_PROFILE_ID, ACCURATE_PROFILE_ID)
        accurate = profile_for(ACCURATE_PROFILE_ID)
        self.assertEqual(accurate.label, "Accurate")
        self.assertEqual(accurate.model_name, "distil-large-v3")
        self.assertEqual(model_name_for_profile(None), "distil-large-v3")

    def test_fast_is_a_smaller_whisper_variant(self):
        fast = profile_for(FAST_PROFILE_ID)
        self.assertEqual(fast.label, "Fast")
        self.assertEqual(fast.model_name, "small")
        self.assertEqual(fast.repo_id, "mlx-community/whisper-small-mlx")
        self.assertIn(FAST_PROFILE_ID, PROFILE_ORDER)
        self.assertIn(ACCURATE_PROFILE_ID, PROFILE_ORDER)
        weights, config = remote_filenames(profile_for(ACCURATE_PROFILE_ID))
        self.assertEqual(weights, "./mlx_models/distil-large-v3/weights.npz")
        self.assertEqual(config, "./mlx_models/distil-large-v3/config.json")

    def test_unknown_profile_normalizes_to_accurate(self):
        self.assertEqual(normalize_profile("parakeet"), ACCURATE_PROFILE_ID)
        self.assertEqual(normalize_profile(None), ACCURATE_PROFILE_ID)
        self.assertEqual(normalize_profile("fast"), FAST_PROFILE_ID)

    def test_download_writes_expected_local_files_and_reports_progress(self):
        with TemporaryDirectory() as temp:
            model_root = Path(temp) / "mlx_models"
            calls = []
            progress = []

            def fake_hub_download(*, repo_id, filename, local_dir, **_kwargs):
                calls.append((repo_id, filename, local_dir))
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / Path(filename).name).write_bytes(b"ok")
                return str(Path(local_dir) / Path(filename).name)

            download_profile(
                profile_for("fast"),
                model_root=model_root,
                on_progress=progress.append,
                download_file=fake_hub_download,
            )
            downloaded = [filename for _repo, filename, _dir in calls]
            self.assertCountEqual(
                downloaded, list(remote_filenames(profile_for("fast")))
            )
            # Config is fetched first so the large weights file can report most of the progress.
            self.assertEqual(downloaded[0], "config.json")
            self.assertGreaterEqual(len(progress), 2)
            self.assertEqual(progress[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
