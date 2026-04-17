#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/../../model/lpr.bin"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[Error] Missing model file: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
python3 main.py --model-path "${MODEL_PATH}" "$@"
