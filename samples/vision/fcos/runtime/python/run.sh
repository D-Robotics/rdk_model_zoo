#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
MODEL_FILE="${MODEL_DIR}/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin"
URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin"

mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_FILE}" ]; then
  wget -O "${MODEL_FILE}" "${URL}"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
