#!/usr/bin/env bash
set -euo pipefail

# The legacy path is a compatibility entrypoint.  Dependencies and model
# files must be prepared explicitly; the canonical script only configures,
# builds, and runs the one maintained C++ implementation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL_DIR="${SCRIPT_DIR}/../../../../../../../samples/vision/paddle_ocr/runtime/cpp"
LEGACY_DATA="${SCRIPT_DIR}/../../test_data"
exec bash "${CANONICAL_DIR}/run.sh" -- \
    --test_image "${LEGACY_DATA}/gt_2322.jpg" \
    --label_file "${LEGACY_DATA}/ppocrv6_dict.txt" \
    --font_path "${LEGACY_DATA}/FangSong.ttf" \
    "$@"
