#!/usr/bin/env bash
set -e

BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO"

MODELS=(
  "yolo11n_detect_bayese_640x640_nv12.bin"
  "yolo11n_seg_bayese_640x640_nv12.bin"
  "yolo11n_pose_bayese_640x640_nv12.bin"
  "yolo11n_cls_detect_bayese_640x640_nv12.bin"
)

for model in "${MODELS[@]}"; do
  if [ ! -f "${model}" ]; then
    wget -O "${model}" "${BASE_URL}/${model}"
  fi
done
