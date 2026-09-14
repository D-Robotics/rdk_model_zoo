#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Platform compatibility check: LaneNet only supports S100
if [[ "$SOC" != "s100" ]]; then
    echo ""
    echo "[WARNING] LaneNet only supports RDK S100 (s100)."
    echo "[WARNING] Current platform: $SOC"
    echo "[WARNING] The model was trained and compiled for S100 BPU."
    echo "[WARNING] Inference results on $SOC may be incorrect or fail."
    echo "[WARNING] Please refer to README.md for platform compatibility details."
    echo ""
fi

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
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

# Model Download (S100 only)
MODEL_PATH="/opt/hobot/model/s100/basic/lanenet256x512.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/lanenet256x512.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# Model Execution
python3 main.py \
    --model-path /opt/hobot/model/s100/basic/lanenet256x512.hbm \
    --test-img ../../test_data/lane.jpg \
    --instance-save-path instance_pred.png \
    --binary-save-path binary_pred.png \
    --priority 0 \
    --bpu-cores 0
