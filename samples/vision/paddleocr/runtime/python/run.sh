#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

DET_MODEL="${MODEL_DIR}/en_PP-OCRv3_det_640x640_nv12.bin"
REC_MODEL="${MODEL_DIR}/en_PP-OCRv3_rec_48x320_rgb.bin"

# Download models if missing
if [ ! -f "${DET_MODEL}" ] || [ ! -f "${REC_MODEL}" ]; then
    bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
