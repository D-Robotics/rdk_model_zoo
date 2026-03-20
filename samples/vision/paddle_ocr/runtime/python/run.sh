#!/bin/bash
set -e

# Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Platform compatibility check: PaddleOCR only supports S100
if [[ "$SOC" != "s100" ]]; then
    echo ""
    echo "[WARNING] PaddleOCR only supports RDK S100 (s100)."
    echo "[WARNING] Current platform: $SOC"
    echo "[WARNING] The model was compiled for S100 BPU."
    echo "[WARNING] Inference results on $SOC may be incorrect or fail."
    echo "[WARNING] Please refer to README.md for platform compatibility details."
    echo ""
fi

# Environment Setup
PYTHON_BIN=python3
PIP_BIN=pip3

REQUIREMENTS=(
  "numpy==1.26.4"
  "opencv-python==4.11.0.86"
  "pyclipper==1.3.0.post6"
  "Pillow==9.0.1"
)

check_and_install() {
  local pkg="$1"
  local name="${pkg%%==*}"
  local version="${pkg##*==}"

  installed_version=$($PIP_BIN show "$name" 2>/dev/null | awk '/^Version:/{print $2}')

  if [[ "$installed_version" == "$version" ]]; then
    echo "$name==$version already installed, skip"
  else
    if [[ -n "$installed_version" ]]; then
      echo "$name version mismatch (installed: $installed_version, need: $version)"
    else
      echo "$name not installed, installing $version"
    fi
    $PIP_BIN install "$name==$version"
  fi
}

for pkg in "${REQUIREMENTS[@]}"; do
  check_and_install "$pkg"
done

# Model Download (S100 only)
DET_MODEL_PATH="/opt/hobot/model/s100/basic/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm"
DET_MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_det_infer-deploy_640x640_nv12.hbm"

REC_MODEL_PATH="/opt/hobot/model/s100/basic/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm"
REC_MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/paddle_ocr/cn_PP-OCRv3_rec_infer-deploy_48x320_rgb.hbm"

echo "Det model  : $DET_MODEL_PATH"
if [[ ! -f "$DET_MODEL_PATH" ]]; then
  echo "Detection model not found, downloading..."
  mkdir -p "$(dirname "$DET_MODEL_PATH")"
  curl -fL "$DET_MODEL_URL" -o "$DET_MODEL_PATH"
  echo "Detection model downloaded successfully"
else
  echo "Detection model already exists, skip download"
fi

echo "Rec model  : $REC_MODEL_PATH"
if [[ ! -f "$REC_MODEL_PATH" ]]; then
  echo "Recognition model not found, downloading..."
  mkdir -p "$(dirname "$REC_MODEL_PATH")"
  curl -fL "$REC_MODEL_URL" -o "$REC_MODEL_PATH"
  echo "Recognition model downloaded successfully"
else
  echo "Recognition model already exists, skip download"
fi

# Model Execution
python3 main.py \
    --det-model-path "$DET_MODEL_PATH" \
    --rec-model-path "$REC_MODEL_PATH" \
    --test-img ../../test_data/gt_2322.jpg \
    --label-file ../../test_data/ppocr_keys_v1.txt \
    --img-save-path result.jpg
