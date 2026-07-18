#!/usr/bin/env python3
"""Terminating probe for Python and native imports inside the app bundle."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Callable

IMPORT_MODULES = (
    "AppKit",
    "ApplicationServices",
    "Quartz",
    "objc",
    "mlx",
    "mlx._reprlib_fix",
    "mlx.core",
    "numpy",
    "numpy._core._multiarray_umath",
    "pyaudio",
    "pyaudio._portaudio",
    "pynput",
    "rumps",
    "lightning_whisper_mlx",
    "fvt.app",
    "fvt.audio",
    "fvt.coordinator",
    "fvt.insertion",
    "fvt.transcription",
    "fvt.ui",
)
SUCCESS_SENTINEL = "FVT_BUNDLE_IMPORT_PROBE_OK"
OPENMP_PROBE_NAME = "numba-openmp"


def run_numba_openmp_probe(
    import_module: Callable[[str], object] | None = None,
) -> None:
    """Execute a deterministic parallel kernel through Numba's public API."""
    importer = import_module or importlib.import_module
    os.environ["NUMBA_THREADING_LAYER"] = "omp"
    os.environ["NUMBA_NUM_THREADS"] = "2"
    numba = importer("numba")
    numpy = importer("numpy")

    @numba.njit(parallel=True, cache=False)
    def parallel_sum(values):
        total = 0
        for index in numba.prange(values.size):
            total += values[index]
        return total

    values = numpy.arange(256, dtype=numpy.int64)
    expected = int(values.sum())
    actual = int(parallel_sum(values))
    if actual != expected:
        raise RuntimeError(f"Numba OpenMP probe returned {actual}, expected {expected}")
    threading_layer = str(numba.threading_layer())
    if threading_layer != "omp":
        raise RuntimeError(
            f"Numba selected {threading_layer!r}, expected the OpenMP backend"
        )


def run_import_probe(
    import_module: Callable[[str], object] | None = None,
    openmp_probe: Callable[[Callable[[str], object]], None] | None = None,
) -> int:
    importer = import_module or importlib.import_module
    try:
        # Run before any higher-level package can import Numba and freeze its
        # threading configuration.
        print(f"FVT_IMPORT_BEGIN {OPENMP_PROBE_NAME}", flush=True)
        (openmp_probe or run_numba_openmp_probe)(importer)
        print(f"FVT_IMPORT_OK {OPENMP_PROBE_NAME}", flush=True)
        for module in IMPORT_MODULES:
            print(f"FVT_IMPORT_BEGIN {module}", flush=True)
            importer(module)
            print(f"FVT_IMPORT_OK {module}", flush=True)
    except BaseException:  # The standalone probe must never enter py2app report_error.
        traceback.print_exc()
        return 1
    print(SUCCESS_SENTINEL, flush=True)
    return 0


if __name__ == "__main__":
    status = run_import_probe()
    sys.stdout.flush()
    sys.stderr.flush()
    # Native libraries can register teardown work that is irrelevant to import
    # validity. Bypass interpreter finalization so this build probe is bounded.
    os._exit(status)
