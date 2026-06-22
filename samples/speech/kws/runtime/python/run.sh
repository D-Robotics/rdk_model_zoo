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
  "paddlepaddle"
  "tqdm"
  "scikit-learn"
  "paddleaudio"
)

check_and_install() {
  local pkg="$1"
  local name="${pkg%%==*}"

  $PIP_BIN show "$name" >/dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    echo "$name already installed, skip"
  else
    echo "$name not installed, installing"
    $PIP_BIN install "$pkg"
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

# Model Download
MODEL_PATH="/opt/hobot/model/${SOC}/basic/kws.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/kws/kws.hbm"

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
    --model-path "$MODEL_PATH" \
    --audio-file ../../test_data/sample.wav \
    --priority 0 \
    --bpu-cores 0
