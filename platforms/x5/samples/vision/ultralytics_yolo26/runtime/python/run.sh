#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

# Download model if missing
if [ ! -f "${MODEL_DIR}/yolo26n_detect_bayese_640x640_nv12.bin" ]; then
    bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
