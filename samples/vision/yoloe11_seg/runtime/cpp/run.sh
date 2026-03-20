#!/bin/bash
set -e

# 1. Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# S600 platform check
if [[ "$SOC" == "s600" ]]; then
  echo "[ERROR] YOLOe11-Seg is not supported on S600 platform. Exiting."
  exit 1
fi

# 2. Environment Setup
PKGS=(
  libgflags-dev
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

# 3. Model Download (S100 only — S600 not supported)
MODEL_PATH="/opt/hobot/model/${SOC}/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  curl -fL "$MODEL_URL" -o "$MODEL_PATH"
  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# 4. Model Compilation
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 5. Quick Run
./yoloe11seg \
    --model_path /opt/hobot/model/${SOC}/basic/yoloe_11s_seg_pf_nashe_640x640_nv12.hbm \
    --test_img ../../../test_data/office_desk.jpg \
    --label_file ../../../test_data/coco_extended.names \
    --score_thres 0.25 \
    --nms_thres 0.7
