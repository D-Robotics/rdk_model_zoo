#!/bin/bash
set -e

# 1. Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# 2. Environment Setup
PKGS=(
  libgflags-dev
)

need_update=false

# Check if there are any missing packages
for pkg in "${PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    need_update=true
    break
  fi
done

# Only update apt index if there are packages to install
if $need_update; then
  echo "Running apt update (packages missing)"
  sudo apt update
fi

# Install missing packages
for pkg in "${PKGS[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "$pkg already installed"
  else
    echo "Installing $pkg"
    sudo apt install -y "$pkg"
  fi
done

# 3. Model Download
MODEL_PATH="/opt/hobot/model/${SOC}/basic/yolov5x_672x672_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  curl -fL "$MODEL_URL" -o "$MODEL_PATH"

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# 4. Model Execution
mkdir -p build && cd build
cmake ..
make -j$(nproc)

./yolov5 \
    --model-path /opt/hobot/model/${SOC}/basic/yolov5x_672x672_nv12.hbm \
    --test-img /app/res/assets/kite.jpg \
    --label-file /app/res/labels/coco_classes.names \
    --score-thres 0.25 \
    --nms-thres 0.45
