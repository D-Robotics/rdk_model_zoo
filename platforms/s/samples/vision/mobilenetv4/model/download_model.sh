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

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_URL_BASE="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/MobileNet"
MODEL_DIR="$(dirname "$0")/${MODEL_SOC}"

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
