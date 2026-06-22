#!/usr/bin/env bash
set -e

SOC=${1:-s100}
VARIANT=${2:-all}

if [ "${SOC}" != "s100" ]; then
  echo "Unsupported SOC: ${SOC}"
  echo "This sample provides the public S100 model artifacts."
  exit 1
fi

MODEL_DIR="./s100"
mkdir -p "${MODEL_DIR}"

download_one() {
  local model_file="$1"
  wget -c -P "${MODEL_DIR}" \
    "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/EfficientNet/${model_file}"
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
