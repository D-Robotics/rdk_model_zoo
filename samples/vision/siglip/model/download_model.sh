#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOC_DIR="${SCRIPT_DIR}/s100"
mkdir -p "${SOC_DIR}"
cd "${SOC_DIR}"

MODEL_NAME=${1:-bpu-siglip-base-patch16-224}
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/SigLIP"
MODEL_FILE="${MODEL_NAME}.hbm"

if [ -f "${MODEL_FILE}" ]; then
  echo "Model already exists: ${MODEL_FILE}"
  exit 0
fi

wget -c "${BASE_URL}/${MODEL_FILE}" -O "${MODEL_FILE}"
