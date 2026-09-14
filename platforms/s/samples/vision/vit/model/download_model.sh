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
    "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ViT/${model_file}"
}

case "${VARIANT}" in
  int8)
    download_one vit_cifar10_batch1_int8.hbm
    ;;
  int16)
    download_one vit_cifar10_batch1_int16.hbm
    ;;
  all)
    download_one vit_cifar10_batch1_int8.hbm
    download_one vit_cifar10_batch1_int16.hbm
    ;;
  *)
    echo "Unsupported ViT variant: ${VARIANT}"
    echo "Supported variants: int8, int16, all"
    exit 1
    ;;
esac
