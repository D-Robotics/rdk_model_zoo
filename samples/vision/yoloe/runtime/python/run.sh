#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

MODEL_FILE="${MODEL_DIR}/yoloe_11s_seg_pf_bayese_640x640_nv12.bin"
CUSTOM_MODEL=0
for arg in "$@"; do
    case "$arg" in
        --model-path|--model-path=*) CUSTOM_MODEL=1 ;;
    esac
done

# Download model if missing
if [ "${CUSTOM_MODEL}" -eq 0 ] && [ ! -f "${MODEL_FILE}" ]; then
    bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
