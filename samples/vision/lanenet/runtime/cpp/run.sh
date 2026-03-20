#!/bin/bash
set -e

# 1. Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# Platform compatibility check: LaneNet only supports S100
if [[ "$SOC" != "s100" ]]; then
    echo ""
    echo "[WARNING] LaneNet only supports RDK S100 (s100)."
    echo "[WARNING] Current platform: $SOC"
    echo "[WARNING] The model was trained and compiled for S100 BPU."
    echo "[WARNING] Inference results on $SOC may be incorrect or fail."
    echo "[WARNING] Please refer to README.md for platform compatibility details."
    echo ""
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

# 3. Model Download (S100 only)
MODEL_PATH="/opt/hobot/model/s100/basic/lanenet256x512.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/lanenet256x512.hbm"

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
./lanenet \
    --model_path /opt/hobot/model/s100/basic/lanenet256x512.hbm \
    --test_img ../../../test_data/lane.jpg \
    --instance_save_path instance_pred.png \
    --binary_save_path binary_pred.png
