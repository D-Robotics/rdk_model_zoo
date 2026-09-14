#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

MODEL_FILE="${MODEL_DIR}/modnet_512x512_rgb.bin"

# Download model if missing
if [ ! -f "${MODEL_FILE}" ]; then
    bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
