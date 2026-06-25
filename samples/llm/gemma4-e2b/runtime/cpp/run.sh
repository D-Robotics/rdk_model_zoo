#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"

SOC=""
if [[ -r /sys/class/boardinfo/soc_name ]]; then
  SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
  echo "SOC         : $SOC"
fi

PKGS=(cmake g++ libopencv-dev cargo)
need_update=false
for pkg in "${PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    need_update=true
    break
  fi
done

if $need_update; then
  echo "Running apt update (packages missing)..."
  sudo apt update
fi

for pkg in "${PKGS[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "$pkg already installed"
  else
    echo "Installing $pkg..."
    sudo apt install -y "$pkg"
  fi
done

echo "GEMMA4_HOME : $GEMMA4_HOME"
GEMMA4_HOME="$GEMMA4_HOME" bash "$SAMPLE_ROOT/model/download_model.sh"

mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

export GEMMA4_HOME
echo
echo "Starting gemma4_chat (Ctrl+C to exit)..."
echo "Try: /image $SAMPLE_ROOT/test_data/image1.jpg"
echo "Then: Describe this image"
echo
exec ./gemma4_chat
