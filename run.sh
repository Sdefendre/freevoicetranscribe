#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
STAMP="$VENV_DIR/.requirements.sha256"

find_python() {
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3; do
        if ! command -v "$candidate" > /dev/null 2>&1; then
            continue
        fi
        if "$candidate" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info < (3, 14) else 1)' > /dev/null 2>&1; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

if [ ! -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$(find_python)" || {
        echo "Error: Python 3.11-3.13 is required." >&2
        exit 1
    }
    echo "Creating the local Python environment with $PYTHON..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

CURRENT_HASH="$(shasum -a 256 "$SCRIPT_DIR/requirements.txt" | awk '{print $1}')"
INSTALLED_HASH="$(test -f "$STAMP" && sed -n '1p' "$STAMP" || true)"
if [ "$CURRENT_HASH" != "$INSTALLED_HASH" ]; then
    echo "Installing pinned runtime dependencies..."
    "$VENV_DIR/bin/python" -m pip install --disable-pip-version-check -r "$SCRIPT_DIR/requirements.txt"
    printf '%s\n' "$CURRENT_HASH" > "$STAMP"
fi

export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV_DIR/bin/python" -m fvt "$@"
