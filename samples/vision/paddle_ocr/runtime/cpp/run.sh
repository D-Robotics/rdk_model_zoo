#!/usr/bin/env bash
set -euo pipefail

# This entrypoint only builds and runs an already-prepared sample.  Model
# downloads and package installation are deliberately explicit user steps;
# see runtime/cpp/README.md.  Model paths default to the board's
# /opt/hobot/model location; pass --det-model-path/--rec-model-path to use a
# locally prepared artifact.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel "${JOBS:-$(nproc 2>/dev/null || echo 1)}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi

# Resolve data from this checkout instead of the caller's current directory;
# an explicitly supplied flag always wins.
DEFAULT_IMAGE="${SAMPLE_DIR}/test_data/s100/gt_2322.jpg"
DEFAULT_VOCAB="${SAMPLE_DIR}/test_data/s100/ppocrv6_dict.txt"
FONT_PATH="${SAMPLE_DIR}/test_data/FangSong.ttf"

has_flag() {
    local name="$1"
    shift
    local arg
    for arg in "$@"; do
        [[ "${arg}" == "${name}" || "${arg}" == "${name}="* ]] && return 0
    done
    return 1
}
check_default_file() {
    local flag="$1"
    local path="$2"
    shift 2
    if ! has_flag "${flag}" "$@"; then
        [[ -f "${path}" ]] || {
            echo "required fixture is missing: ${path}" >&2
            exit 2
        }
    fi
}
check_default_file --test-img "${DEFAULT_IMAGE}" "$@"
check_default_file --vocabulary-path "${DEFAULT_VOCAB}" "$@"
check_default_file --font-path "${FONT_PATH}" "$@"

NATIVE_ARGS=("$@")
if ! has_flag --test-img "${NATIVE_ARGS[@]}"; then
    NATIVE_ARGS+=(--test-img "${DEFAULT_IMAGE}")
fi
if ! has_flag --vocabulary-path "${NATIVE_ARGS[@]}"; then
    NATIVE_ARGS+=(--vocabulary-path "${DEFAULT_VOCAB}")
fi
if ! has_flag --font-path "${NATIVE_ARGS[@]}"; then
    NATIVE_ARGS+=(--font-path "${FONT_PATH}")
fi
exec "${BUILD_DIR}/paddle_ocr" "${NATIVE_ARGS[@]}"
