#!/bin/bash
set -e

# Read S100 board information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
if [[ -r /sys/class/boardinfo/board_type ]]; then
  BOARD_TYPE=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/board_type)
else
  BOARD_TYPE="$SOC"
fi
echo "SOC        : $SOC"
echo "Board type : $BOARD_TYPE"

# Model suffix differs by platform: S100 uses nashe; S100P uses nashm.
MODEL_MARCH="nash-e"
MODEL_SUFFIX="nashe"
if [[ "$SOC" == "s100p" || "$BOARD_TYPE" == *"p"* ]]; then
  MODEL_MARCH="nash-m"
  MODEL_SUFFIX="nashm"
fi
echo "Model march: $MODEL_MARCH"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "scipy==1.15.3"
)

check_and_install() {
  local pkg="$1"
  local name="${pkg%%==*}"
  local version="${pkg##*==}"

  installed_version=$($PIP_BIN show "$name" 2>/dev/null | awk '/^Version:/{print $2}')

  if [[ "$installed_version" == "$version" ]]; then
    echo "$name==$version already installed, skip"
  else
    if [[ -n "$installed_version" ]]; then
      echo "$name version mismatch (installed: $installed_version, need: $version)"
    else
      echo "$name not installed, installing $version"
    fi
    $PIP_BIN install "$name==$version"
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

# Task selection (default: detect)
TASK="${1:-detect}"

# Model path and URL per task
case "$TASK" in
  detect)
    MODEL_FILE="yolo26n_detect_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  seg)
    MODEL_FILE="yolo26n_seg_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  pose)
    MODEL_FILE="yolo26n_pose_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  cls)
    MODEL_FILE="yolo26n_cls_${MODEL_SUFFIX}_224x224_nv12.hbm"
    ;;
  obb)
    MODEL_FILE="yolo26n_obb_${MODEL_SUFFIX}_640x640_nv12.hbm"
    ;;
  *)
    echo "Unknown task: $TASK. Use: detect, seg, pose, cls, obb"
    exit 1
    ;;
esac

MODEL_PATH="../../model/${MODEL_MARCH}/${MODEL_FILE}"

echo "Task       : $TASK"
echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."
  (cd ../../model && bash download_model.sh)
else
  echo "Model already exists, skip download"
fi

# Run inference
python main.py \
    --task "$TASK" \
    --model-path "$MODEL_PATH" \
    --test-img ../../test_data/bus.jpg \
    --label-file ../../test_data/coco_classes.names \
    --img-save-path result.jpg \
    --priority 0 \
    --bpu-cores 0 \
    --score-thres 0.25 \
    --nms-thres 0.45
