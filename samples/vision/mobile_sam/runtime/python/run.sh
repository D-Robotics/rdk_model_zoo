#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
ENCODER_MODEL="${MODEL_DIR}/mobile_sam_image_encoder_norm_512x512_allint16.bin"
DECODER_MODEL="${MODEL_DIR}/mobile_sam_decoder_512_box_default.bin"

if [ ! -f "${ENCODER_MODEL}" ] || [ ! -f "${DECODER_MODEL}" ]; then
  bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
