#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy>=1.26.4"
  "opencv-python>=4.11.0.86"
  "scipy>=1.15.3"
  "hbm_runtime>=0.1.0.post1"
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

# Model Download
# Default model: efficientnet_lite0
MODEL_NAME="efficientnet_lite0"
MODEL_RES="224x224"
MODEL_FILE="${MODEL_NAME}_${MODEL_RES}_nv12.hbm"
MODEL_PATH="../../model/${MODEL_FILE}"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/EfficientNet/${MODEL_FILE}"

echo "Model path : $MODEL_PATH"
echo "Model Url: $MODEL_URL"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading default model..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Model Execution
python3 main.py \
    --model-path "$MODEL_PATH" \
    --test-img ../../../../../datasets/imagenet/asset/scottish_deerhound.JPEG \
    --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
    --topk 5 \
    --priority 0 \
    --bpu-cores 0
