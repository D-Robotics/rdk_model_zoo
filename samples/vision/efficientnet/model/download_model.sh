#!/bin/bash
set -e

# Resolve target SoC: explicit CLI arg wins, otherwise read /sys/class/boardinfo/soc_name.
# Only s100 and s600 prebuilt HBM files are published; anything else (s100p,
# (null), unknown) falls back to the S100 build.
if [[ -n "${1:-}" ]]; then
  SOC_RAW="$(echo "$1" | tr 'A-Z' 'a-z')"
else
  SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
fi
SOC="${SOC_RAW:-s100}"
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

VARIANT=${2:-all}

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/EfficientNet"
OUTPUT_DIR="$(dirname "$0")/${MODEL_SOC}"
mkdir -p "${OUTPUT_DIR}"

download_one() {
  local model_file="$1"
  local output_path="${OUTPUT_DIR}/${model_file}"
  local url="${MODEL_BASE_URL}/${model_file}"

  echo "Downloading ${model_file} ..."
  wget -c "$url" -O "$output_path"
}

case "${VARIANT}" in
  lite0)
    download_one efficientnet_lite0_224x224_nv12.hbm
    ;;
  lite1)
    download_one efficientnet_lite1_240x240_nv12.hbm
    ;;
  lite2)
    download_one efficientnet_lite2_260x260_nv12.hbm
    ;;
  lite3)
    download_one efficientnet_lite3_300x300_nv12.hbm
    ;;
  lite4)
    download_one efficientnet_lite4_380x380_nv12.hbm
    ;;
  all)
    download_one efficientnet_lite0_224x224_nv12.hbm
    download_one efficientnet_lite1_240x240_nv12.hbm
    download_one efficientnet_lite2_260x260_nv12.hbm
    download_one efficientnet_lite3_300x300_nv12.hbm
    download_one efficientnet_lite4_380x380_nv12.hbm
    ;;
  *)
    echo "Unsupported EfficientNet variant: ${VARIANT}"
    echo "Supported variants: lite0, lite1, lite2, lite3, lite4, all"
    exit 1
    ;;
esac