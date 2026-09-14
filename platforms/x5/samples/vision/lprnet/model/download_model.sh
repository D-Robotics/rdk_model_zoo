#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FILE="${SCRIPT_DIR}/lpr.bin"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/LPRNet/lpr.bin"

if [ ! -f "${MODEL_FILE}" ]; then
  echo "[Info] Downloading LPRNet model from: ${MODEL_URL}"
  wget -O "${MODEL_FILE}" "${MODEL_URL}"
fi

echo "[Info] Model ready: ${MODEL_FILE}"