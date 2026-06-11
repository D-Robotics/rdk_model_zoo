#!/usr/bin/env bash
set -e

SOC=${1:-s100}

if [ "${SOC}" != "s100" ]; then
  echo "Unsupported SOC: ${SOC}"
  echo "This sample provides the public S100 model artifact."
  exit 1
fi

MODEL_DIR="./s100"
mkdir -p "${MODEL_DIR}"

wget -c -P "${MODEL_DIR}" \
  https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ResNet/resnet152_224x224_nv12.hbm
