#!/usr/bin/env bash
# Published X5 ConvNeXt bins. Default variant is atto.
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_atto_224x224_nv12.bin
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_nano_224x224_nv12.bin
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_pico_224x224_nv12.bin
#   https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ConvNeXt_femto_224x224_nv12.bin
# download.py reads the URL from docs/release/x5/models.yaml. Pass nano, pico, or femto as $2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-x5}"
VARIANT="${2:-}"

if [[ -n "$VARIANT" ]]; then
  exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --output-dir "${SCRIPT_DIR}"
fi
exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --output-dir "${SCRIPT_DIR}"
