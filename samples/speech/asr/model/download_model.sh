#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_PATH="/opt/hobot/model/${SOC}/basic/asr.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/asr/asr.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi
