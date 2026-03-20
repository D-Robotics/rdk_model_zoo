#!/bin/bash
set -e

# 1. Read SOC information
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

# 2. Environment Setup
PKGS=(
  libgflags-dev
  libpolyclipping-dev
  libfreetype6-dev
)

need_update=false

for pkg in "${PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    need_update=true
    break
  fi
done

if $need_update; then
  echo "Running apt update (packages missing)"
  sudo apt update
fi

for pkg in "${PKGS[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "$pkg already installed"
  else
    echo "Installing $pkg"
    sudo apt install -y "$pkg"
  fi
done

# 3. Model Download (S100 only)
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

# 4. Model Compilation
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 5. Quick Run
./paddle_ocr \
    --det_model_path "$DET_MODEL_PATH" \
    --rec_model_path "$REC_MODEL_PATH" \
    --test_image ../../../test_data/gt_2322.jpg \
    --label_file ../../../test_data/ppocr_keys_v1.txt \
    --font_path ../../../test_data/FangSong.ttf \
    --img_save_path result.jpg
