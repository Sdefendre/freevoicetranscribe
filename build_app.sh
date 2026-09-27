#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_VENV="${FVT_BUILD_VENV:-$PROJECT_DIR/.build-venv}"
BUILD_LOCK="$PROJECT_DIR/requirements-build.lock"
BUILD_STAMP="$BUILD_VENV/.fvt-build-lock.sha256"
BUILD_ROOT="$PROJECT_DIR/build/release"
FAILED_BUILD_ROOT="$PROJECT_DIR/build/failed-release"
STAGING_DIST="$BUILD_ROOT/dist"
STAGING_APP="$STAGING_DIST/FreeVoiceTranscribe.app"
FINAL_DIST="$PROJECT_DIR/dist"
FINAL_APP="$FINAL_DIST/FreeVoiceTranscribe.app"
BUILD_COMPLETE=0
BACKUP_ROOT=""

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [ "$BUILD_COMPLETE" -ne 1 ]; then
        if [ "${FVT_RETAIN_FAILED_STAGING:-0}" = "1" ] && [ -d "$BUILD_ROOT" ]; then
            rm -rf "$FAILED_BUILD_ROOT"
            mv "$BUILD_ROOT" "$FAILED_BUILD_ROOT"
            echo "Retained failed staging for diagnostics: $FAILED_BUILD_ROOT" >&2
        else
            rm -rf "$BUILD_ROOT"
        fi
        if [ -n "$BACKUP_ROOT" ] && [ -d "$BACKUP_ROOT/FreeVoiceTranscribe.app" ]; then
            rm -rf "$FINAL_APP"
            mv "$BACKUP_ROOT/FreeVoiceTranscribe.app" "$FINAL_APP"
        fi
        echo "Build failed; no partial app was published to dist/." >&2
    else
        rm -rf "$BUILD_ROOT"
    fi
    if [ -n "$BACKUP_ROOT" ]; then
        rmdir "$BACKUP_ROOT" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Error: the app must be built on an Apple Silicon Mac." >&2
    exit 1
fi

if ! command -v portaudio-config > /dev/null 2>&1 \
    && { ! command -v brew > /dev/null 2>&1 || ! brew list portaudio > /dev/null 2>&1; }; then
    echo "Error: PortAudio is required. Install it with: brew install portaudio" >&2
    exit 1
fi

validate_python() {
    "$1" "$PROJECT_DIR/scripts/validate_build_python.py"
}

select_python() {
    local candidate
    local resolved
    if [ -n "${PYTHON:-}" ]; then
        if ! resolved="$(command -v "$PYTHON" 2>/dev/null)"; then
            echo "Error: requested build Python was not found: $PYTHON" >&2
            return 1
        fi
        if ! validate_python "$resolved" >&2; then
            echo "Error: PYTHON must name a framework-capable arm64 Python 3.11." >&2
            return 1
        fi
        printf '%s\n' "$resolved"
        return 0
    fi

    for candidate in \
        /opt/homebrew/bin/python3.11 \
        /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
        python3.11; do
        if ! resolved="$(command -v "$candidate" 2>/dev/null)"; then
            continue
        fi
        if validate_python "$resolved" > /dev/null 2>&1; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done

    echo "Error: no framework-capable arm64 Python 3.11 was found." >&2
    echo "Install Homebrew Python 3.11 or set PYTHON to a compatible interpreter." >&2
    return 1
}

BUILD_PYTHON="$(select_python)"
echo "Using build Python: $BUILD_PYTHON"

LOCK_HASH="$(shasum -a 256 "$BUILD_LOCK" | awk '{print $1}')"
INSTALLED_HASH="$(test -f "$BUILD_STAMP" && sed -n '1p' "$BUILD_STAMP" || true)"
if [ -x "$BUILD_VENV/bin/python" ]; then
    if ! validate_python "$BUILD_VENV/bin/python" > /dev/null 2>&1 \
        || [ "$LOCK_HASH" != "$INSTALLED_HASH" ]; then
        echo "Recreating the build environment for the selected Python and lock file..."
        rm -rf "$BUILD_VENV"
    fi
fi

if [ ! -x "$BUILD_VENV/bin/python" ]; then
    "$BUILD_PYTHON" -m venv "$BUILD_VENV"
fi
validate_python "$BUILD_VENV/bin/python" > /dev/null

"$BUILD_VENV/bin/python" -m pip install --disable-pip-version-check \
    --requirement "$BUILD_LOCK"
printf '%s\n' "$LOCK_HASH" > "$BUILD_STAMP"
"$BUILD_VENV/bin/python" -m pip check

rm -rf "$BUILD_ROOT" "$PROJECT_DIR/.build-assets"
mkdir -p "$PROJECT_DIR/.build-assets/AppIcon.iconset" "$STAGING_DIST"

"$BUILD_VENV/bin/python" "$PROJECT_DIR/generate_icon.py" \
    "$PROJECT_DIR/.build-assets/icon_512.png"

for size in 16 32 64 128 256 512; do
    sips -z "$size" "$size" "$PROJECT_DIR/.build-assets/icon_512.png" \
        --out "$PROJECT_DIR/.build-assets/AppIcon.iconset/icon_${size}x${size}.png" \
        > /dev/null
done
for size in 16 32 128 256; do
    double=$((size * 2))
    cp "$PROJECT_DIR/.build-assets/AppIcon.iconset/icon_${double}x${double}.png" \
        "$PROJECT_DIR/.build-assets/AppIcon.iconset/icon_${size}x${size}@2x.png"
done
sips -z 1024 1024 "$PROJECT_DIR/.build-assets/icon_512.png" \
    --out "$PROJECT_DIR/.build-assets/AppIcon.iconset/icon_512x512@2x.png" \
    > /dev/null
iconutil -c icns "$PROJECT_DIR/.build-assets/AppIcon.iconset" \
    -o "$PROJECT_DIR/.build-assets/AppIcon.icns"

cd "$PROJECT_DIR"
"$BUILD_VENV/bin/python" setup.py py2app \
    --bdist-base "$BUILD_ROOT/bdist" \
    --dist-dir "$STAGING_DIST"

# py2app includes the MLX extension but does not discover the adjacent runtime
# directory that its @loader_path/lib rpath requires. Preserve that exact
# layout, including libmlx, libjaccl, and the Metal shader library.
MLX_RUNTIME_SOURCE="$BUILD_VENV/lib/python3.11/site-packages/mlx/lib"
MLX_RUNTIME_DESTINATION="$STAGING_APP/Contents/Resources/lib/python3.11/lib-dynload/mlx/lib"
if [ ! -f "$MLX_RUNTIME_SOURCE/libmlx.dylib" ] \
    || [ ! -f "$MLX_RUNTIME_SOURCE/libjaccl.dylib" ] \
    || [ ! -f "$MLX_RUNTIME_SOURCE/mlx.metallib" ]; then
    echo "Error: the locked MLX runtime files are incomplete." >&2
    exit 1
fi
mkdir -p "$MLX_RUNTIME_DESTINATION"
cp -R "$MLX_RUNTIME_SOURCE/." "$MLX_RUNTIME_DESTINATION/"

# Numba's macOS wheel references OpenMP through @rpath but does not provide an
# LC_RPATH of its own. Source the runtime from the pinned Torch wheel, copy it
# into Frameworks, and make the bundled extension's load command deterministic.
OPENMP_SOURCE="$BUILD_VENV/lib/python3.11/site-packages/torch/lib/libomp.dylib"
OPENMP_EXTENSION="$STAGING_APP/Contents/Resources/lib/python3.11/numba/np/ufunc/omppool.cpython-311-darwin.so"
OPENMP_DESTINATION="$STAGING_APP/Contents/Frameworks/libomp.dylib"
if [ ! -f "$OPENMP_SOURCE" ] || [ ! -f "$OPENMP_EXTENSION" ]; then
    echo "Error: the locked OpenMP runtime or bundled Numba extension is missing." >&2
    exit 1
fi
cp "$OPENMP_SOURCE" "$OPENMP_DESTINATION"
install_name_tool -id @rpath/libomp.dylib "$OPENMP_DESTINATION"
install_name_tool -change \
    @rpath/libomp.dylib \
    @executable_path/../Frameworks/libomp.dylib \
    "$OPENMP_EXTENSION"

if [ -n "${CODESIGN_IDENTITY:-}" ]; then
    # install_name_tool invalidates the embedded signatures on both relocated
    # OpenMP binaries. Sign modified nested code from the inside out because
    # --deep does not repair extension modules stored below Resources.
    codesign --force --options runtime --timestamp \
        --sign "$CODESIGN_IDENTITY" "$OPENMP_DESTINATION"
    codesign --force --options runtime --timestamp \
        --sign "$CODESIGN_IDENTITY" "$OPENMP_EXTENSION"
    codesign --force --deep --options runtime --timestamp \
        --sign "$CODESIGN_IDENTITY" "$STAGING_APP"
    SIGNING_DESCRIPTION="Developer ID identity: $CODESIGN_IDENTITY"
else
    codesign --force --sign - "$OPENMP_DESTINATION"
    codesign --force --sign - "$OPENMP_EXTENSION"
    codesign --force --deep --sign - "$STAGING_APP"
    SIGNING_DESCRIPTION="ad-hoc local signature"
fi

codesign --verify --deep --strict --verbose=2 "$STAGING_APP"
"$BUILD_VENV/bin/python" scripts/verify_bundle.py \
    --launch-import-smoke \
    --launch-liveness-smoke \
    "$STAGING_APP"

if spctl -a -vv -t exec "$STAGING_APP"; then
    echo "Gatekeeper assessment passed."
else
    echo "Gatekeeper did not accept this local artifact."
    echo "This is expected for ad-hoc builds and Developer ID builds that are not notarized."
fi

mkdir -p "$FINAL_DIST"
BACKUP_ROOT="$(mktemp -d "$FINAL_DIST/.previous-build.XXXXXX")"
if [ -e "$FINAL_APP" ]; then
    mv "$FINAL_APP" "$BACKUP_ROOT/FreeVoiceTranscribe.app"
fi
mv "$STAGING_APP" "$FINAL_APP"
BUILD_COMPLETE=1
rm -rf "$BACKUP_ROOT"
BACKUP_ROOT=""

echo "Built and verified: $FINAL_APP"
echo "Code signing: $SIGNING_DESCRIPTION"
if [ -z "${CODESIGN_IDENTITY:-}" ]; then
    echo "This ad-hoc artifact is for local testing only."
fi
echo "Apple distribution still requires Developer ID credentials and notarization."
