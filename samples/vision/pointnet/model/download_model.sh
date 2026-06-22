#!/usr/bin/env bash
set -e

SOC=${1:-s100}

if [ "${SOC}" != "s100" ]; then
  echo "Unsupported SOC: ${SOC}"
  echo "This sample provides the public S100 PointNet model artifact only."
  exit 1
fi

MODEL_DIR="./s100"
MODEL_FILE="pointnet.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/PointNet/${MODEL_FILE}"

mkdir -p "${MODEL_DIR}"

if [ -s "${MODEL_DIR}/${MODEL_FILE}" ]; then
  echo "${MODEL_DIR}/${MODEL_FILE} already exists, skip download."
  exit 0
fi

echo "Downloading ${MODEL_FILE}..."
wget -c "${MODEL_URL}" -O "${MODEL_DIR}/${MODEL_FILE}"
echo "Download complete."
