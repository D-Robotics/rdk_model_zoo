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
mkdir -p "${MODEL_DIR}"

wget -c -P "${MODEL_DIR}" \
  "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ResNet/resnet152_224x224_nv12.hbm"
