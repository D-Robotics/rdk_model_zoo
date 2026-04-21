#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${SCRIPT_DIR}/lpr.bin"

if [ ! -f "${MODEL_FILE}" ]; then
  echo "[Error] lpr.bin is expected to be bundled with this sample." >&2
  exit 1
fi

echo "[Info] Using bundled model: ${MODEL_FILE}"
