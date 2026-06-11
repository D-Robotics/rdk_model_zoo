#!/bin/bash
set -e

# Detect S100 board variant. S100 uses nash-e; S100P uses nash-m.
if [[ -r /sys/class/boardinfo/soc_name ]]; then
  SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
else
  SOC="s100"
fi
if [[ -r /sys/class/boardinfo/board_type ]]; then
  BOARD_TYPE=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/board_type)
else
  BOARD_TYPE="$SOC"
fi

MODEL_MARCH="nash-e"
MODEL_SUFFIX="nashe"
if [[ "$SOC" == "s100p" || "$BOARD_TYPE" == *"p"* ]]; then
  MODEL_MARCH="nash-m"
  MODEL_SUFFIX="nashm"
fi

echo "Detected SoC: ${SOC}"
echo "Board type  : ${BOARD_TYPE}"
echo "Model march : ${MODEL_MARCH}"
echo "Model suffix: ${MODEL_SUFFIX}"

BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Ultralytics_YOLO_OE_3.7.0/${MODEL_MARCH}"

MODELS=(
  "yolo26n_detect_${MODEL_SUFFIX}_640x640_nv12.hbm"
  "yolo26n_seg_${MODEL_SUFFIX}_640x640_nv12.hbm"
  "yolo26n_pose_${MODEL_SUFFIX}_640x640_nv12.hbm"
  "yolo26n_cls_${MODEL_SUFFIX}_224x224_nv12.hbm"
  "yolo26n_obb_${MODEL_SUFFIX}_640x640_nv12.hbm"
)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TARGET_DIR="${SCRIPT_DIR}/${MODEL_MARCH}"
mkdir -p "${TARGET_DIR}"

for model in "${MODELS[@]}"; do
  TARGET="${TARGET_DIR}/${model}"
  if [[ -f "$TARGET" ]]; then
    echo "${model} already exists, skip"
  else
    echo "Downloading ${model}..."
    wget -c "${BASE_URL}/${model}" -O "${TARGET}" || echo "Failed to download ${model}"
  fi
done

echo "Download complete. Models are stored in ${TARGET_DIR}."
