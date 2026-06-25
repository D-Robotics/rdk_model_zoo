#!/bin/bash
set -e

# 1. Read SOC information; route s600 to its own published HBM, fall back to
# the S100 build for s100 / s100p / (null) / unknown.
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

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
MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/mobilenetv2_224x224_nv12.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/MobileNet/mobilenetv2_224x224_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading..."

  mkdir -p "$(dirname "$MODEL_PATH")"

  if command -v wget &>/dev/null; then
    wget -q "$MODEL_URL" -O "$MODEL_PATH"
  elif command -v curl &>/dev/null; then
    curl -fL "$MODEL_URL" -o "$MODEL_PATH"
  else
    echo "ERROR: neither wget nor curl found" >&2
    exit 1
  fi

  echo "Model downloaded successfully"
else
  echo "Model already exists, skip download"
fi

# 4. Model Compilation
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 5. Quick Run
./mobilenetv2 \
  --model_path "$MODEL_PATH" \
  --test_img   ../../../test_data/zebra_cls.jpg \
  --label_file ../../../test_data/imagenet1000_labels.txt \
  --top_k 5
