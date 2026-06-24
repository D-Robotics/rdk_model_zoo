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
DET_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
REC_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"

echo "Det model URL : $DET_MODEL_URL"
echo "Rec model URL : $REC_MODEL_URL"
echo ""

wget "$DET_MODEL_URL"
wget "$REC_MODEL_URL"
