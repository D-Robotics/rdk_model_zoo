#!/usr/bin/env bash
# Default S100/S600 files are the nashe/nashp HBMs selected by variant.
# rdk_s archive HBMs (pass the manifest filename as $3; URLs are in
# docs/release/s/models.yaml):
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet18_224x224_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet18_224x224_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet50_224x224_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet50_224x224_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152_224x224_nv12.hbm
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ResNet/resnet152_224x224_nv12.hbm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-x5}"
VARIANT="${2:-18}"
FILENAME="${3:-}"

if [[ -n "$FILENAME" ]]; then
  exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --filename "${FILENAME}" --output-dir "${SCRIPT_DIR}"
fi
exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --output-dir "${SCRIPT_DIR}"
