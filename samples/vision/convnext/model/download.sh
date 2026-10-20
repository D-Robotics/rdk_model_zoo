#!/usr/bin/env bash
# The only published ConvNeXt artifact is X5 ConvNeXt Atto
# (ConvNeXt_atto_224x224_nv12.bin). download.py accepts variant atto only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-x5}"
VARIANT="${2:-}"

if [[ -n "$VARIANT" ]]; then
  exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --variant "${VARIANT}" --output-dir "${SCRIPT_DIR}"
fi
exec python3 "${SCRIPT_DIR}/download.py" --target "${TARGET}" --output-dir "${SCRIPT_DIR}"
