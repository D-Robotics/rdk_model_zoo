#!/bin/bash
set -e

SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
MODEL_DIR="/opt/hobot/model/${SOC}/basic"
MODEL_FILE="${MODEL_DIR}/kws.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/kws/kws.hbm"

if [[ -f "$MODEL_FILE" ]]; then
  echo "Model already exists: ${MODEL_FILE}"
  exit 0
fi

echo "Downloading KWS model..."
mkdir -p "$MODEL_DIR"
curl -fL "$MODEL_URL" -o "$MODEL_FILE"
echo "Download complete: ${MODEL_FILE}"
