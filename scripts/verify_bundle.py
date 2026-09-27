#!/usr/bin/env python3
"""Strict static and isolated-import checks for the py2app bundle."""

from __future__ import annotations

import argparse
import os
import plistlib
import selectors
import signal
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

IMPORT_PROBE_NAME = "bundle_import_probe"
IMPORT_SUCCESS_SENTINEL = b"FVT_BUNDLE_IMPORT_PROBE_OK"
APP_READY_SENTINEL = b"FVT_APP_READY"
REQUIRED_PYTHON_MODULES = (
    "fvt.app",
    "fvt.coordinator",
    "lightning_whisper_mlx.lightning",
    "mlx",
    "mlx._reprlib_fix",
    "numba",
    "numpy",
    "pyaudio",
    "pynput._util.darwin",
    "pynput.keyboard._darwin",
    "pynput.mouse._darwin",
)


class BundleVerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise BundleVerificationError(message)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def resolve_bundle_reference(reference: str, executable: Path, contents: Path) -> Path:
    if reference.startswith("@executable_path/"):
        return executable.parent / reference.removeprefix("@executable_path/")
    if reference.startswith("@loader_path/"):
        return executable.parent / reference.removeprefix("@loader_path/")
    path = Path(reference)
    if path.is_absolute():
        return path
    _fail(f"unsupported runtime reference: {reference}")


def _python_inventory(resources: Path) -> set[str]:
    inventory: set[str] = set()
    for path in resources.rglob("*"):
        if path.is_file() and not path.is_symlink():
            inventory.add(path.relative_to(resources).as_posix().lower())
    for archive in resources.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as stream:
                inventory.update(name.lower() for name in stream.namelist())
        except zipfile.BadZipFile:
            _fail(f"invalid Python archive: {archive.relative_to(resources)}")
    return inventory


def module_present(inventory: set[str], module: str) -> bool:
    stem = module.replace(".", "/").lower()
    candidates = (
        f"{stem}.py",
        f"{stem}.pyc",
        f"{stem}/__init__.py",
        f"{stem}/__init__.pyc",
    )
    return any(
        any(entry.endswith(candidate) for candidate in candidates)
        for entry in inventory
    )


def _mach_o_candidates(
    contents: Path, executable: Path, runtimes: list[Path]
) -> set[Path]:
    candidates = {executable, *runtimes}
    for suffix in ("*.so", "*.dylib", "*.bundle"):
        candidates.update(contents.rglob(suffix))
    frameworks = contents / "Frameworks"
    if frameworks.is_dir():
        for path in frameworks.rglob("*"):
            if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK):
                candidates.add(path)
    return {path for path in candidates if path.is_file() and not path.is_symlink()}


def _dependency_target_exists(
    dependency: str,
    *,
    binary: Path,
    executable: Path,
    contents: Path,
) -> bool:
    if dependency.startswith("@loader_path/"):
        target = binary.parent / dependency.removeprefix("@loader_path/")
        return target.resolve().is_file()
    if dependency.startswith("@executable_path/"):
        target = executable.parent / dependency.removeprefix("@executable_path/")
        return target.resolve().is_file()
    if dependency.startswith("@rpath/"):
        suffix = dependency.removeprefix("@rpath/")
        return any(path.is_file() for path in contents.rglob(Path(suffix).name))
    if dependency.startswith("/System/Library/") or dependency.startswith("/usr/lib/"):
        return True
    if dependency.startswith("/"):
        return False
    return True


def _otool_dependencies(output: str) -> list[str]:
    dependencies: list[str] = []
    for line in output.splitlines():
        # Dependency records are indented. Universal binaries can emit one
        # unindented file/architecture header per slice, not just one header.
        if not line[:1].isspace():
            continue
        dependency = line.strip().split(" (compatibility", 1)[0]
        if dependency:
            dependencies.append(dependency)
    return dependencies


def _verify_mach_o_files(
    contents: Path,
    executable: Path,
    runtimes: list[Path],
) -> list[Path]:
    mach_o_files: list[Path] = []
    for path in sorted(_mach_o_candidates(contents, executable, runtimes)):
        kind = _run(["file", "-b", str(path)])
        if kind.returncode != 0:
            _fail(f"could not inspect {path.relative_to(contents)}")
        if "Mach-O" not in kind.stdout:
            if path in {executable, *runtimes}:
                _fail(
                    f"required runtime file is not Mach-O: {path.relative_to(contents)}"
                )
            continue
        if "arm64" not in kind.stdout:
            _fail(f"Mach-O does not contain arm64: {path.relative_to(contents)}")
        mach_o_files.append(path)

        links = _run(["otool", "-L", str(path)])
        if links.returncode != 0:
            _fail(f"otool failed for {path.relative_to(contents)}")
        for dependency in _otool_dependencies(links.stdout):
            if not _dependency_target_exists(
                dependency,
                binary=path,
                executable=executable,
                contents=contents,
            ):
                _fail(
                    f"unresolved or external dependency in "
                    f"{path.relative_to(contents)}: {dependency}"
                )

    if not mach_o_files:
        _fail("no Mach-O files were found")
    return mach_o_files


def _require_native_modules(mach_o_files: list[Path], contents: Path) -> None:
    relative = [path.relative_to(contents).as_posix().lower() for path in mach_o_files]
    requirements = {
        "PyAudio": lambda value: "_portaudio" in Path(value).name,
        "PyObjC": lambda value: Path(value).name.startswith("_objc."),
        "NumPy": lambda value: "_multiarray_umath" in Path(value).name,
        "MLX": lambda value: (
            "/mlx/" in f"/{value}" and Path(value).name.startswith("core.")
        ),
        "Numba OpenMP": lambda value: Path(value).name.startswith("omppool."),
    }
    for name, predicate in requirements.items():
        if not any(predicate(value) for value in relative):
            _fail(f"required native module is missing: {name}")


def _signature_failure_detail(signature: subprocess.CompletedProcess[str]) -> str:
    lines = signature.stderr.strip().splitlines()
    if lines:
        return lines[-1]
    return f"codesign exited with status {signature.returncode}"


def _verify_code_signatures(
    app: Path, mach_o_files: list[Path], contents: Path
) -> None:
    # codesign --deep can accept an outer bundle while a Mach-O extension under
    # Resources still has a stale embedded signature. dyld then kills the app
    # when that extension is loaded, so verify every unpacked image directly.
    for path in mach_o_files:
        signature = _run(["codesign", "--verify", "--strict", "--verbose=2", str(path)])
        if signature.returncode != 0:
            _fail(
                "strict code-sign verification failed for "
                f"{path.relative_to(contents)}: "
                f"{_signature_failure_detail(signature)}"
            )

    signature = _run(
        ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)]
    )
    if signature.returncode != 0:
        _fail(
            "strict code-sign verification failed: "
            f"{_signature_failure_detail(signature)}"
        )


def _isolated_environment(temp: str, **extra: str) -> dict[str, str]:
    environment = {
        "HOME": temp,
        "LANG": "en_US.UTF-8",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": temp,
        "FVT_APP_SUPPORT": str(Path(temp) / "support"),
        "FVT_CACHE_DIR": str(Path(temp) / "cache"),
        "FVT_LOG_DIR": str(Path(temp) / "logs"),
    }
    environment.update(extra)
    return environment


def _terminate_process_group(
    process: subprocess.Popen[bytes], timeout: float = 5.0
) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=timeout)

    # The py2app launcher is not expected to fork, but ensure a failed smoke
    # cannot leave a descendant from its isolated process group behind.
    # The leader has been reaped. Its former group may already be gone or
    # inaccessible; do not turn a successful smoke into a cleanup failure.
    # Signal directly to avoid a check-then-kill race.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _capture_until(
    process: subprocess.Popen[bytes], sentinel: bytes, timeout: float
) -> tuple[bool, bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        raise AssertionError("smoke process pipes were not configured")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    deadline = time.monotonic() + timeout
    found = False
    try:
        while time.monotonic() < deadline:
            for key, _events in selector.select(
                timeout=min(0.1, max(0.0, deadline - time.monotonic()))
            ):
                chunk = os.read(key.fileobj.fileno(), 65_536)
                if chunk:
                    output[key.data].extend(chunk)
                    if key.data == "stdout" and sentinel in output["stdout"]:
                        found = True
                        return found, bytes(output["stdout"]), bytes(output["stderr"])
                else:
                    selector.unregister(key.fileobj)
            if process.poll() is not None and not selector.get_map():
                break
    finally:
        selector.close()
    return found, bytes(output["stdout"]), bytes(output["stderr"])


def _remaining_output(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    stdout, stderr = process.communicate(timeout=5)
    return stdout or b"", stderr or b""


def _output_detail(stdout: bytes, stderr: bytes) -> str:
    text = (stderr or stdout).decode("utf-8", errors="replace").strip()
    return text.splitlines()[-1] if text else "no diagnostic output"


def _process_exit_detail(returncode: int | None) -> str:
    if returncode is None:
        return "process exit status unavailable"
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"signal {signal_number}"
        return f"terminated by {signal_name} ({signal_number})"
    return f"exited with status {returncode}"


def _probe_failure_detail(
    stdout: bytes, stderr: bytes, returncode: int | None = None
) -> str:
    active_module: str | None = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("FVT_IMPORT_BEGIN "):
            active_module = line.removeprefix("FVT_IMPORT_BEGIN ").strip()
        elif line.startswith("FVT_IMPORT_OK "):
            completed = line.removeprefix("FVT_IMPORT_OK ").strip()
            if completed == active_module:
                active_module = None
    error = (
        _output_detail(b"", stderr)
        if stderr.strip()
        else _process_exit_detail(returncode)
    )
    if active_module is not None:
        return f"{active_module}: {error}"
    return _output_detail(stdout, stderr)


def _run_import_smoke(probe: Path, timeout: float = 60.0) -> None:
    with tempfile.TemporaryDirectory(prefix="fvt-bundle-smoke-") as temp:
        process = subprocess.Popen(
            [str(probe)],
            cwd=temp,
            env=_isolated_environment(temp),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            found, stdout, stderr = _capture_until(
                process, IMPORT_SUCCESS_SENTINEL, timeout
            )
            if not found:
                if process.poll() is None:
                    _terminate_process_group(process)
                remaining_stdout, remaining_stderr = _remaining_output(process)
                stdout += remaining_stdout
                stderr += remaining_stderr
                _fail(
                    "bundled-import probe failed: "
                    f"{_probe_failure_detail(stdout, stderr, process.returncode)}"
                )
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                _fail("bundled-import probe reported success but did not terminate")
            remaining_stdout, remaining_stderr = _remaining_output(process)
            stdout += remaining_stdout
            stderr += remaining_stderr
        finally:
            _terminate_process_group(process)
        if process.returncode != 0:
            _fail(
                "bundled-import probe failed: "
                f"{_probe_failure_detail(stdout, stderr, process.returncode)}"
            )


def _run_launch_liveness_smoke(
    executable: Path, timeout: float = 30.0, liveness_duration: float = 1.0
) -> None:
    with tempfile.TemporaryDirectory(prefix="fvt-launch-smoke-") as temp:
        process = subprocess.Popen(
            [str(executable)],
            cwd=temp,
            env=_isolated_environment(temp, FVT_LAUNCH_SMOKE="1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout = b""
        stderr = b""
        failure: str | None = None
        try:
            found, stdout, stderr = _capture_until(process, APP_READY_SENTINEL, timeout)
            if not found:
                failure = "application launch smoke failed"
            else:
                # The marker is emitted immediately before entering the menu-bar
                # loop. It must remain live rather than exit like the import probe.
                time.sleep(liveness_duration)
                if process.poll() is not None:
                    failure = "application exited after readiness"
        finally:
            _terminate_process_group(process)
            remaining_stdout, remaining_stderr = _remaining_output(process)
            stdout += remaining_stdout
            stderr += remaining_stderr

        if failure is not None:
            _fail(f"{failure}: {_output_detail(stdout, stderr)}")

        error_output = stderr.decode("utf-8", errors="replace")
        fatal_markers = (
            "Traceback (most recent call last)",
            "Application startup failed",
            "encountered a fatal error",
            "An uncaught exception was raised",
        )
        if any(marker in error_output for marker in fatal_markers):
            _fail(
                f"application entered py2app report_error: {_output_detail(b'', stderr)}"
            )


def verify_bundle(
    app: Path,
    *,
    verify_signature: bool = True,
    launch_import_smoke: bool = False,
    launch_liveness_smoke: bool = False,
) -> int:
    app = app.resolve()
    contents = app / "Contents"
    resources = contents / "Resources"
    plist_path = contents / "Info.plist"
    if not app.is_dir() or not plist_path.is_file():
        _fail("the app bundle or Info.plist is missing")
    with plist_path.open("rb") as stream:
        plist = plistlib.load(stream)

    if plist.get("CFBundleIdentifier") != "com.stevedefendre.freevoicetranscribe":
        _fail("unexpected bundle identifier")
    if plist.get("LSUIElement") is not True:
        _fail("the menu-bar app must have LSUIElement=true")
    if plist.get("LSMinimumSystemVersion") != "14.0":
        _fail("the minimum macOS version must be 14.0")
    if not plist.get("NSMicrophoneUsageDescription"):
        _fail("microphone usage description is missing")

    executable = contents / "MacOS" / str(plist.get("CFBundleExecutable", ""))
    if not executable.is_file() or not os.access(executable, os.X_OK):
        _fail("bundle executable is missing or not executable")
    if (resources / ".venv").exists():
        _fail("a copied virtualenv was found")
    if (resources / "mlx_models").exists():
        _fail("models must live in Application Support, not the signed bundle")

    for path in contents.rglob("*"):
        if path.is_symlink():
            resolved = path.resolve()
            if not _inside(resolved, app):
                _fail(f"external symlink: {path.relative_to(app)} -> {resolved}")

    runtime_references = plist.get("PyRuntimeLocations")
    if not isinstance(runtime_references, list) or not runtime_references:
        _fail("PyRuntimeLocations is missing")
    runtimes: list[Path] = []
    for reference in runtime_references:
        if not isinstance(reference, str):
            _fail("PyRuntimeLocations contains a non-string entry")
        runtime = resolve_bundle_reference(reference, executable, contents).resolve()
        if not _inside(runtime, app):
            _fail(f"Python runtime resolves outside the app: {reference}")
        if not runtime.is_file():
            _fail(f"Python runtime is missing: {reference}")
        runtimes.append(runtime)

    import_probe = contents / "MacOS" / IMPORT_PROBE_NAME
    if not import_probe.is_file() or not os.access(import_probe, os.X_OK):
        _fail("the bundled-import probe is missing or not executable")

    archives = [path for path in resources.rglob("python*.zip") if path.stat().st_size]
    if not archives:
        _fail("the bundled Python standard-library archive is missing")
    inventory = _python_inventory(resources)
    for module in REQUIRED_PYTHON_MODULES:
        if not module_present(inventory, module):
            _fail(f"required Python module is missing: {module}")

    mach_o_files = _verify_mach_o_files(contents, executable, [*runtimes, import_probe])
    _require_native_modules(mach_o_files, contents)

    if verify_signature:
        _verify_code_signatures(app, mach_o_files, contents)
    if launch_import_smoke:
        _run_import_smoke(import_probe)
    if launch_liveness_smoke:
        _run_launch_liveness_smoke(executable)
    return len(mach_o_files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-signature",
        action="store_true",
        help="skip only the signature check (for isolated fixture tests)",
    )
    parser.add_argument(
        "--launch-import-smoke",
        action="store_true",
        help="run the terminating bundled Python/native import probe",
    )
    parser.add_argument(
        "--launch-liveness-smoke",
        action="store_true",
        help="launch the normal app and require it to enter its menu-bar loop",
    )
    parser.add_argument("app", type=Path)
    arguments = parser.parse_args()
    try:
        checked = verify_bundle(
            arguments.app,
            verify_signature=not arguments.skip_signature,
            launch_import_smoke=arguments.launch_import_smoke,
            launch_liveness_smoke=arguments.launch_liveness_smoke,
        )
    except BundleVerificationError as exc:
        parser.exit(1, f"Bundle verification failed: {exc}\n")
    print(
        f"Bundle verification passed ({checked} arm64 Mach-O files checked): "
        f"{arguments.app.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
