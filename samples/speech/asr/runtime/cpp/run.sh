#!/bin/bash
set -e

# 1. Read SOC information
SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
echo "SOC        : $SOC"

# 2. Environment Setup
PKGS=(
  libgflags-dev
  libsndfile1-dev
  libsamplerate0-dev
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

# 3. Model Download
MODEL_PATH="/opt/hobot/model/${SOC}/basic/asr.hbm"
MODEL_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${SOC}/asr/asr.hbm"

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
./asr \
    --model_path "$MODEL_PATH" \
    --test_sound ../../../test_data/chi_sound.wav \
    --vocab_file ../../../test_data/vocab.json
