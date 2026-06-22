#!/usr/bin/env bash
set -e

SOC="${1:-s100}"
MODEL_FILE="mobilenetv1_224x224_nv12.hbm"

echo "SOC        : $SOC"

if [[ "$SOC" != "s100" ]]; then
  echo "MobileNetV1 public HBM is currently provided for S100 only."
  echo "Use: bash download_model.sh s100"
  exit 1
fi

MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet/${MODEL_FILE}"
OUTPUT_DIR="$(dirname "$0")/s100"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_FILE}"

mkdir -p "$OUTPUT_DIR"

echo "Model URL  : $MODEL_URL"
echo "Output     : $OUTPUT_PATH"

if [[ -f "$OUTPUT_PATH" ]]; then
  echo "Model already exists, skip download"
  exit 0
fi

wget -c "$MODEL_URL" -O "$OUTPUT_PATH"
echo "Model downloaded successfully"
