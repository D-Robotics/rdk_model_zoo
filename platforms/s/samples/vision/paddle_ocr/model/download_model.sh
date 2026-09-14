#!/bin/bash
set -e

# Read SOC information (strip "(null)" / whitespace → default s100)
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"

# Map SOC to the corresponding model archive directory.
# Only S100 and S600 have pre-quantized PP-OCRv6 hbm models on the archive;
# any other SoC (e.g. S100P) falls back to the S100 build.
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/paddle_ocr"
DET_MODEL_FILE="PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
REC_MODEL_FILE="PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"

# Unified install path: aligned with runtime/{python,cpp}/run.sh and the
# default --det_model_path / --rec_model_path in main.py / main.cpp so all
# entry points read from the same location.
OUTPUT_DIR="/opt/hobot/model/${MODEL_SOC}/basic"
DET_OUTPUT="${OUTPUT_DIR}/${DET_MODEL_FILE}"
REC_OUTPUT="${OUTPUT_DIR}/${REC_MODEL_FILE}"

mkdir -p "$OUTPUT_DIR"

echo "Det model URL : ${MODEL_BASE_URL}/${DET_MODEL_FILE}"
echo "Rec model URL : ${MODEL_BASE_URL}/${REC_MODEL_FILE}"
echo "Output dir    : $OUTPUT_DIR"
echo ""

if [[ -f "$DET_OUTPUT" ]]; then
  echo "${DET_MODEL_FILE} already exists, skip download"
else
  wget -c "${MODEL_BASE_URL}/${DET_MODEL_FILE}" -O "$DET_OUTPUT"
fi

if [[ -f "$REC_OUTPUT" ]]; then
  echo "${REC_MODEL_FILE} already exists, skip download"
else
  wget -c "${MODEL_BASE_URL}/${REC_MODEL_FILE}" -O "$REC_OUTPUT"
fi