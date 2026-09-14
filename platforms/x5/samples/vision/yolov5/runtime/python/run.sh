#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
MODEL_FILE="${MODEL_DIR}/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"

mkdir -p "${MODEL_DIR}"

if [ ! -f "${MODEL_FILE}" ]; then
  wget -O "${MODEL_FILE}" "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
