#!/usr/bin/env bash
set -euo pipefail

# This entrypoint only builds and runs an already-prepared sample.  Model
# downloads and package installation are deliberately explicit user steps;
# see runtime/cpp/README.md.  Pass normal executable arguments after ``--``:
#
#   ./run.sh -- --det_model_path /opt/... --rec_model_path /opt/...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel "${JOBS:-$(nproc 2>/dev/null || echo 1)}"

if [[ "${1:-}" == "--" ]]; then
    shift
fi

# Resolve data from this checkout instead of the caller's current directory.
# Defaults come first so gflags' normal last-value-wins behavior lets an
# explicitly supplied user flag override any fixture path.
DEFAULT_IMAGE="${SAMPLE_DIR}/test_data/s100/gt_2322.jpg"
DEFAULT_LABEL="${SAMPLE_DIR}/test_data/s100/ppocrv6_dict.txt"
FONT_PATH="${REPO_ROOT}/platforms/s/samples/vision/paddle_ocr/test_data/FangSong.ttf"
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
check_default_file --test_image "${DEFAULT_IMAGE}" "$@"
check_default_file --label_file "${DEFAULT_LABEL}" "$@"
check_default_file --font_path "${FONT_PATH}" "$@"
DEFAULT_ARGS=(
    --test_image "${DEFAULT_IMAGE}"
    --label_file "${DEFAULT_LABEL}"
    --font_path "${FONT_PATH}"
)
exec "${BUILD_DIR}/paddle_ocr" "${DEFAULT_ARGS[@]}" "$@"
