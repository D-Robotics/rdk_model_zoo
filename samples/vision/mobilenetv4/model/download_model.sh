#!/bin/bash
set -e

SOC="${1:-s100}"

if [[ "$SOC" != "s100" ]]; then
  echo "Only S100 MobileNetV4 HBM files are available in the public archive."
  echo "Requested SoC: $SOC"
  exit 1
fi

MODEL_URL_BASE="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/MobileNet"
MODEL_DIR="./s100"

mkdir -p "$MODEL_DIR"

download_one() {
  local model_name="$1"
  local output_path="${MODEL_DIR}/${model_name}.hbm"
  local url="${MODEL_URL_BASE}/${model_name}.hbm"

  if [[ -f "$output_path" ]]; then
    echo "${output_path} already exists, skip"
    return
  fi

  echo "Downloading ${model_name}.hbm..."
  wget -c "$url" -O "$output_path"
}

download_one "mobilenetv4_small_224x224_nv12"
download_one "mobilenetv4_medium_256x256_nv12"

echo "All MobileNetV4 models downloaded to ${MODEL_DIR}."
