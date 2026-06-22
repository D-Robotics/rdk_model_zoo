#!/usr/bin/env bash
set -e

SOC=${1:-s100}

# Supported SoC -> model directory (also doubles as URL sub-path).
#   s100 -> rdk_s100/ResNet
#   s600 -> rdk_s600/ResNet
case "${SOC}" in
  s100|s600) ;;
  *)
    echo "Unsupported SOC: ${SOC}"
    echo "Available: s100, s600"
    exit 1
    ;;
esac

MODEL_DIR="./${SOC}"
MODEL_NAME="resnet152_224x224_nv12"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ResNet/${MODEL_NAME}.hbm"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}.hbm"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
  echo "${MODEL_PATH} already exists, skip"
  exit 0
fi

echo "Downloading ${MODEL_NAME}.hbm for ${SOC}..."
wget -c "${MODEL_URL}" -O "${MODEL_PATH}"
echo "Downloaded to ${MODEL_PATH}"
