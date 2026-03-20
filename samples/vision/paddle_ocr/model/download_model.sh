#!/bin/bash
set -e

# PaddleOCR models only support RDK S100 platform
DET_MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm"
REC_MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm"

echo "Det model URL : $DET_MODEL_URL"
echo "Rec model URL : $REC_MODEL_URL"
echo ""
echo "[NOTE] PaddleOCR models only support RDK S100. These models are NOT compatible with RDK S600."
echo ""

wget "$DET_MODEL_URL"
wget "$REC_MODEL_URL"
