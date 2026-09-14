#!/usr/bin/env bash
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

MODEL_FILE="mobilenetv1_224x224_nv12.hbm"

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/MobileNet/${MODEL_FILE}"
OUTPUT_DIR="$(dirname "$0")/${MODEL_SOC}"
OUTPUT_PATH="${OUTPUT_DIR}/${MODEL_FILE}"

mkdir -p "$OUTPUT_DIR"

echo "Model URL     : $MODEL_URL"
echo "Output        : $OUTPUT_PATH"

if [[ -f "$OUTPUT_PATH" ]]; then
  echo "Model already exists, skip download"
  exit 0
fi

wget -c "$MODEL_URL" -O "$OUTPUT_PATH"
echo "Model downloaded successfully"
