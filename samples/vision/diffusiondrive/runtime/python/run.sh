#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${SAMPLE_DIR}/model/s600/diffusiondrive_r34_256x1024_s600.hbm}"

if [ ! -f "${MODEL_PATH}" ]; then
  MODEL_PATH="${MODEL_PATH}" bash "${SAMPLE_DIR}/model/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py --model-path "${MODEL_PATH}" "$@"
