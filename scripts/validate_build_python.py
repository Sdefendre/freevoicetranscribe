#!/usr/bin/env python3
"""Reject Python builds that cannot produce a portable py2app bundle."""

from __future__ import annotations

import platform
import sys
import sysconfig
import zlib
from pathlib import Path


def validation_errors(
    *,
    version: tuple[int, int],
    machine: str,
    framework: str,
    zlib_file: str | None,
    zlib_exists: bool,
) -> list[str]:
    errors: list[str] = []
    if version != (3, 11):
        errors.append(f"Python 3.11 is required, found {version[0]}.{version[1]}")
    if machine != "arm64":
        errors.append(f"arm64 is required, found {machine or 'unknown architecture'}")
    if framework != "Python":
        errors.append("a macOS framework build of Python is required")
    if not zlib_file or not zlib_exists:
        errors.append("zlib must be a loadable extension with a real file path")
    return errors


def current_validation_errors() -> list[str]:
    zlib_file = getattr(zlib, "__file__", None)
    return validation_errors(
        version=sys.version_info[:2],
        machine=platform.machine(),
        framework=str(sysconfig.get_config_var("PYTHONFRAMEWORK") or ""),
        zlib_file=zlib_file,
        zlib_exists=bool(zlib_file and Path(zlib_file).is_file()),
    )


def main() -> int:
    errors = current_validation_errors()
    if errors:
        print(f"Rejected build Python: {sys.executable}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Compatible build Python: {sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
