#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
ENCODER_MODEL="${MODEL_DIR}/efficient_sam_vitt_encoder_512x512_default_none.bin"
DECODER_MODEL="${MODEL_DIR}/efficient_sam_vitt_decoder_fixedprompt_512_default.bin"

if [ ! -f "${ENCODER_MODEL}" ] || [ ! -f "${DECODER_MODEL}" ]; then
  bash "${MODEL_DIR}/download_model.sh"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
