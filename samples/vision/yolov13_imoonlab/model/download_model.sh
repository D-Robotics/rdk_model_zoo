#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_DIR="${SCRIPT_DIR}/s100"
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/iMoonLab_YOLOv13"

MODELS=(
  "yolo13n_detect_nashe_640x640_nv12.hbm"
  "yolo13s_detect_nashe_640x640_nv12.hbm"
  "yolo13l_detect_nashe_640x640_nv12.hbm"
  "yolo13x_detect_nashe_640x640_nv12.hbm"
)

mkdir -p "${TARGET_DIR}"

for model in "${MODELS[@]}"; do
  target="${TARGET_DIR}/${model}"
  if [[ -f "${target}" ]]; then
    echo "${model} already exists, skip"
    continue
  fi
  echo "Downloading ${model}..."
  wget -c "${BASE_URL}/${model}" -O "${target}"
done

echo "Download complete. Models are stored in ${TARGET_DIR}."
