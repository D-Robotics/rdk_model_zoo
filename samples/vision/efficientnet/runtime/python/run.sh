#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

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

# Model Download
# Local model path configuration
MODEL_PATH="../../model/efficientnet_lite0_224x224_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/efficientnet_lite0_224x224_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  # Use curl to download the model
  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Test data check (Optional: could download if missing)
TEST_IMG="../../test_data/Scottish_deerhound.JPEG"
LABEL_FILE="../../../../../datasets/imagenet/imagenet1000_clsidx_to_labels.txt"

# Model Execution
$PYTHON_BIN main.py \
    --model-path "$MODEL_PATH" \
    --test-img "$TEST_IMG" \
    --label-file "$LABEL_FILE" \
    --topk 5 \
    --resize-type 1 \
    --priority 0 \
    --bpu-cores 0
