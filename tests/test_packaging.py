import os
import plistlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.bundle_import_probe import (
    IMPORT_MODULES,
    run_import_probe,
    run_numba_openmp_probe,
)
from scripts.validate_build_python import validation_errors
from scripts.verify_bundle import (
    APP_READY_SENTINEL,
    IMPORT_SUCCESS_SENTINEL,
    REQUIRED_PYTHON_MODULES,
    BundleVerificationError,
    _otool_dependencies,
    _probe_failure_detail,
    _process_exit_detail,
    _python_inventory,
    _run_import_smoke,
    _run_launch_liveness_smoke,
    _verify_code_signatures,
    module_present,
    verify_bundle,
)


class BuildPythonValidationTests(unittest.TestCase):
    def test_rejects_static_python_without_extension_backed_zlib(self):
        errors = validation_errors(
            version=(3, 11),
            machine="arm64",
            framework="",
            zlib_file=None,
            zlib_exists=False,
        )
        self.assertIn("a macOS framework build of Python is required", errors)
        self.assertIn("zlib must be a loadable extension with a real file path", errors)

    def test_accepts_framework_arm64_python_311(self):
        self.assertEqual(
            validation_errors(
                version=(3, 11),
                machine="arm64",
                framework="Python",
                zlib_file="/Frameworks/Python/zlib.so",
                zlib_exists=True,
            ),
            [],
        )


class BundleVerifierTests(unittest.TestCase):
    def _partial_bundle(self, root: Path) -> Path:
        app = root / "FreeVoiceTranscribe.app"
        contents = app / "Contents"
        executable = contents / "MacOS" / "FreeVoiceTranscribe"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"not a complete executable")
        executable.chmod(0o755)
        plist = {
            "CFBundleExecutable": executable.name,
            "CFBundleIdentifier": "com.stevedefendre.freevoicetranscribe",
            "LSMinimumSystemVersion": "14.0",
            "LSUIElement": True,
            "NSMicrophoneUsageDescription": "Microphone access",
            "PyRuntimeLocations": [
                "@executable_path/../Frameworks/libpython3.11.dylib"
            ],
        }
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(plist, stream)
        return app

    def test_rejects_partial_bundle_with_missing_python_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            app = self._partial_bundle(Path(temp))
            with self.assertRaisesRegex(BundleVerificationError, "runtime is missing"):
                verify_bundle(app, verify_signature=False)

    def test_module_inventory_accepts_source_or_bytecode_entries(self):
        inventory = {
            "lib/python311.zip/fvt/app.pyc",
            "lib/python3.11/site-packages/numpy/__init__.py",
        }
        self.assertTrue(module_present(inventory, "fvt.app"))
        self.assertTrue(module_present(inventory, "numpy"))
        self.assertFalse(module_present(inventory, "pyaudio"))

    def test_invalid_python_archive_is_rejected_before_false_success(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "python311.zip"
            archive.write_bytes(b"not a zip")
            with self.assertRaisesRegex(
                BundleVerificationError, "invalid Python archive"
            ):
                _python_inventory(Path(temp))

    def test_otool_parser_ignores_each_universal_architecture_header(self):
        output = """/tmp/extension.so (architecture x86_64):
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1351.0.0)
/tmp/extension.so (architecture arm64):
\t@rpath/libexample.dylib (compatibility version 1.0.0, current version 1.0.0)
"""
        self.assertEqual(
            _otool_dependencies(output),
            ["/usr/lib/libSystem.B.dylib", "@rpath/libexample.dylib"],
        )

    def test_import_probe_covers_python_and_explicit_native_modules(self):
        required_native = {
            "objc",
            "mlx._reprlib_fix",
            "mlx.core",
            "numpy._core._multiarray_umath",
            "pyaudio._portaudio",
        }
        self.assertTrue(required_native.issubset(IMPORT_MODULES))
        self.assertNotIn("numba.np.ufunc.omppool", IMPORT_MODULES)
        self.assertIn("mlx._reprlib_fix", REQUIRED_PYTHON_MODULES)
        self.assertTrue(
            {
                "pynput._util.darwin",
                "pynput.keyboard._darwin",
                "pynput.mouse._darwin",
            }.issubset(REQUIRED_PYTHON_MODULES)
        )
        imported = []
        openmp_calls = []
        importer = imported.append
        self.assertEqual(
            run_import_probe(
                importer,
                lambda importer: openmp_calls.append(importer),
            ),
            0,
        )
        self.assertEqual(imported, list(IMPORT_MODULES))
        self.assertEqual(openmp_calls, [importer])

    def test_numba_openmp_probe_uses_public_parallel_api(self):
        import numpy

        class FakeNumba:
            prange = range

            @staticmethod
            def njit(*, parallel, cache):
                self.assertTrue(parallel)
                self.assertFalse(cache)
                return lambda function: function

            @staticmethod
            def threading_layer():
                return "omp"

        modules = {"numba": FakeNumba, "numpy": numpy}
        with patch.dict(os.environ, {}, clear=False):
            run_numba_openmp_probe(modules.__getitem__)
            self.assertEqual(os.environ["NUMBA_THREADING_LAYER"], "omp")
            self.assertEqual(os.environ["NUMBA_NUM_THREADS"], "2")

    def test_import_smoke_runs_a_terminating_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp) / "bundle_import_probe"
            probe.write_text(
                f"#!/bin/sh\nprintf '{IMPORT_SUCCESS_SENTINEL.decode()}\\n'\n"
            )
            probe.chmod(0o755)
            _run_import_smoke(probe, timeout=1.0)

    def test_import_smoke_reports_last_incremental_probe_line(self):
        with tempfile.TemporaryDirectory() as temp:
            probe = Path(temp) / "bundle_import_probe"
            probe.write_text(
                "#!/bin/sh\nprintf 'FVT_IMPORT_BEGIN mlx.core\\n'\nsleep 5\n"
            )
            probe.chmod(0o755)
            with self.assertRaisesRegex(BundleVerificationError, "mlx.core"):
                _run_import_smoke(probe, timeout=0.5)

    def test_probe_failure_detail_preserves_module_and_native_error(self):
        self.assertEqual(
            _probe_failure_detail(
                b"FVT_IMPORT_BEGIN mlx.core\n",
                b"ImportError: native initialization failed\n",
            ),
            "mlx.core: ImportError: native initialization failed",
        )

    def test_probe_failure_detail_reports_signal_without_stderr(self):
        self.assertEqual(
            _probe_failure_detail(
                b"FVT_IMPORT_BEGIN numba-openmp\n",
                b"",
                -9,
            ),
            "numba-openmp: terminated by SIGKILL (9)",
        )
        self.assertEqual(_process_exit_detail(70), "exited with status 70")

    def test_failed_staging_retention_is_opt_in_and_outside_dist(self):
        script = (Path(__file__).resolve().parents[1] / "build_app.sh").read_text()
        self.assertIn("FVT_RETAIN_FAILED_STAGING:-0", script)
        self.assertIn("build/failed-release", script)
        self.assertNotIn("dist/failed-release", script)

    def test_relocated_openmp_binaries_are_signed_before_outer_app(self):
        script = (Path(__file__).resolve().parents[1] / "build_app.sh").read_text()
        extension_sign = script.index('codesign --force --sign - "$OPENMP_EXTENSION"')
        outer_sign = script.index('codesign --force --deep --sign - "$STAGING_APP"')
        self.assertLess(extension_sign, outer_sign)

    def test_individual_signature_gate_rejects_invalid_resource_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp) / "FreeVoiceTranscribe.app"
            contents = app / "Contents"
            extension = contents / "Resources" / "omppool.so"
            extension.parent.mkdir(parents=True)
            extension.write_bytes(b"Mach-O fixture")
            invalid = type(
                "Result",
                (),
                {
                    "returncode": 1,
                    "stderr": "omppool.so: invalid signature\n",
                },
            )()
            with patch("scripts.verify_bundle._run", return_value=invalid):
                with self.assertRaisesRegex(
                    BundleVerificationError,
                    r"Resources/omppool\.so: omppool\.so: invalid signature",
                ):
                    _verify_code_signatures(app, [extension], contents)

    def test_launch_smoke_requires_readiness_then_liveness(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "FreeVoiceTranscribe"
            executable.write_text(
                f"#!/bin/sh\nprintf '{APP_READY_SENTINEL.decode()}\\n'\nsleep 5\n"
            )
            executable.chmod(0o755)
            _run_launch_liveness_smoke(
                executable,
                timeout=1.0,
                liveness_duration=0.05,
            )

    def test_launch_smoke_rejects_immediate_exit_after_readiness(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / "FreeVoiceTranscribe"
            executable.write_text(
                f"#!/bin/sh\nprintf '{APP_READY_SENTINEL.decode()}\\n'\n"
            )
            executable.chmod(0o755)
            with self.assertRaisesRegex(
                BundleVerificationError, "exited after readiness"
            ):
                _run_launch_liveness_smoke(
                    executable,
                    timeout=1.0,
                    liveness_duration=0.05,
                )


if __name__ == "__main__":
    unittest.main()
