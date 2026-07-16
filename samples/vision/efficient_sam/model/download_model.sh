#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER_MODEL="${SCRIPT_DIR}/efficient_sam_vitt_encoder_512x512_default_none.bin"
DECODER_MODEL="${SCRIPT_DIR}/efficient_sam_vitt_decoder_fixedprompt_512_default.bin"
ENCODER_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficient_sam/efficient_sam_vitt_encoder_512x512_default_none.bin"
DECODER_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficient_sam/efficient_sam_vitt_decoder_fixedprompt_512_default.bin"

mkdir -p "${SCRIPT_DIR}"

if [ ! -f "${ENCODER_MODEL}" ]; then
  echo "[Info] Downloading EfficientSAM-Tiny encoder from: ${ENCODER_URL}"
  wget -O "${ENCODER_MODEL}" "${ENCODER_URL}"
fi

if [ ! -f "${DECODER_MODEL}" ]; then
  echo "[Info] Downloading EfficientSAM-Tiny decoder from: ${DECODER_URL}"
  wget -O "${DECODER_MODEL}" "${DECODER_URL}"
fi

echo "[Info] EfficientSAM-Tiny models ready in: ${SCRIPT_DIR}"
