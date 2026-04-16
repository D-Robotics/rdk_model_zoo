#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../../model"
TASK="${1:-detect}"

mkdir -p "${MODEL_DIR}"

if [ "${TASK}" = "seg" ]; then
  MODEL_FILE="${MODEL_DIR}/yolo11n_seg_bayese_640x640_nv12.bin"
  URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_seg_bayese_640x640_nv12.bin"
elif [ "${TASK}" = "pose" ]; then
  MODEL_FILE="${MODEL_DIR}/yolo11n_pose_bayese_640x640_nv12.bin"
  URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_pose_bayese_640x640_nv12.bin"
elif [ "${TASK}" = "cls" ]; then
  MODEL_FILE="${MODEL_DIR}/yolo11n_cls_detect_bayese_640x640_nv12.bin"
  URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_cls_detect_bayese_640x640_nv12.bin"
else
  MODEL_FILE="${MODEL_DIR}/yolo11n_detect_bayese_640x640_nv12.bin"
  URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/ultralytics_YOLO/yolo11n_detect_bayese_640x640_nv12.bin"
fi

if [ ! -f "${MODEL_FILE}" ]; then
  wget -O "${MODEL_FILE}" "${URL}"
fi

cd "${SCRIPT_DIR}"
if [ "$#" -gt 0 ]; then
  shift
fi
python3 main.py --task "${TASK}" "$@"
