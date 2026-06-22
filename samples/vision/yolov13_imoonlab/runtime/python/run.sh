#!/bin/bash
set -e

MODEL_PATH="../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm"
TEST_IMAGE="../../test_data/kite.jpg"
LABEL_FILE="../../test_data/coco_classes.names"

echo "Model path : $MODEL_PATH"

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

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading reference models..."
  (cd ../../model && bash download_model.sh)
else
  echo "Model already exists, skip download"
fi

# Model Execution
$PYTHON_BIN main.py \
    --model-path "$MODEL_PATH" \
    --test-img "$TEST_IMAGE" \
    --label-file "$LABEL_FILE" \
    --img-save-path result.jpg \
    --priority 0 \
    --bpu-cores 0 \
    --score-thres 0.25 \
    --nms-thres 0.45
