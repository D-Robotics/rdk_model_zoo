#!/bin/bash
set -e

SOC="${1:-s100}"

if [[ "$SOC" != "s100" ]]; then
  echo "Only the S100 ResNet50 HBM file is available in this sample."
  echo "Requested SoC: $SOC"
  exit 1
fi

MODEL_DIR="./s100"
MODEL_NAME="resnet50_224x224_nv12"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/${MODEL_NAME}.hbm"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}.hbm"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_PATH" ]]; then
  echo "${MODEL_PATH} already exists, skip"
  exit 0
fi

echo "Downloading ${MODEL_NAME}.hbm..."
wget -c "$MODEL_URL" -O "$MODEL_PATH"
echo "Downloaded to ${MODEL_PATH}"
