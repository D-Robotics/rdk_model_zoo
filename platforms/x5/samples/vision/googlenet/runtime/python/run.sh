#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"

MODEL_PATH="${MODEL_DIR}/googlenet_224x224_nv12.bin"

# Download model if missing
if [ ! -f "${MODEL_PATH}" ]; then
    bash "${MODEL_DIR}/download.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
