#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}"

mkdir -p "${MODEL_DIR}"

wget -O "${MODEL_DIR}/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin" \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fcos_efficientnetb0_detect_512x512_bayese_nv12.bin
wget -O "${MODEL_DIR}/fcos_efficientnetb2_detect_768x768_bayese_nv12.bin" \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fcos_efficientnetb2_detect_768x768_bayese_nv12.bin
wget -O "${MODEL_DIR}/fcos_efficientnetb3_detect_896x896_bayese_nv12.bin" \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/fcos_efficientnetb3_detect_896x896_bayese_nv12.bin
