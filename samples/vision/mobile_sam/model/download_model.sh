#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENCODER_MODEL="${SCRIPT_DIR}/mobile_sam_image_encoder_norm_512x512_allint16.bin"
DECODER_MODEL="${SCRIPT_DIR}/mobile_sam_decoder_512_box_default.bin"
ENCODER_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_image_encoder_norm_512x512_allint16.bin"
DECODER_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_decoder_512_box_default.bin"

mkdir -p "${SCRIPT_DIR}"

if [ ! -f "${ENCODER_MODEL}" ]; then
  echo "[Info] Downloading MobileSAM encoder from: ${ENCODER_URL}"
  wget -O "${ENCODER_MODEL}" "${ENCODER_URL}"
fi

if [ ! -f "${DECODER_MODEL}" ]; then
  echo "[Info] Downloading MobileSAM decoder from: ${DECODER_URL}"
  wget -O "${DECODER_MODEL}" "${DECODER_URL}"
fi

echo "[Info] MobileSAM models ready in: ${SCRIPT_DIR}"
