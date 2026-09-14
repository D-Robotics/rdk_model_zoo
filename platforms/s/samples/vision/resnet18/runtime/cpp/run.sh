#!/bin/bash
set -e

# 1. Environment Setup
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

# 2. Model Download
MODEL_PATH="../../model/s100/resnet18_224x224_nv12.hbm"

echo "Model path : $MODEL_PATH"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found, downloading to sample-local model directory..."
  (cd ../../model && bash download_model.sh s100)
else
  echo "Model already exists, skip download"
fi

# 3. Model Compilation
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 4. Quick Run
./resnet18 \
  --model_path ../../../model/s100/resnet18_224x224_nv12.hbm \
  --test_img   ../../../test_data/zebra_cls.jpg \
  --label_file ../../../../../../datasets/imagenet/imagenet_classes.names \
  --top_k 5
