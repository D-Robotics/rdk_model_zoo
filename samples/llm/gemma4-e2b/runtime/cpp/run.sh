#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"

# Optional proxy for git/cargo/cmake downloads (export before running):
#   export HTTP_PROXY=http://192.168.66.115:1082 HTTPS_PROXY=$HTTP_PROXY
if [[ -n "${HTTP_PROXY:-}" ]]; then
  export http_proxy="${HTTP_PROXY}"
  export https_proxy="${HTTPS_PROXY:-${HTTP_PROXY}}"
fi

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
bash "$SAMPLE_ROOT/third_party/install_tokenizers_cpp.sh"

# tokenizers-cpp sets CARGO_TARGET=aarch64-unknown-linux-gnu on aarch64 even
# for native builds, so cc-rs looks for aarch64-unknown-linux-gnu-gcc. Stock
# Ubuntu provides aarch64-linux-gnu-gcc (no "unknown"). Create symlinks so
# the Rust build finds the compiler without patching upstream.
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
  for tool in gcc g++ ar; do
    if ! command -v "aarch64-unknown-linux-gnu-$tool" >/dev/null 2>&1; then
      ln -sf "$(command -v "aarch64-linux-gnu-$tool" || command -v "$tool")" \
             /usr/local/bin/"aarch64-unknown-linux-gnu-$tool" 2>/dev/null || true
    fi
  done
fi

mkdir -p build
cd build
cmake ..
make -j"$(nproc)"

export GEMMA4_HOME
echo
echo "Starting main (Ctrl+C to exit)..."
echo "Try: /image $SAMPLE_ROOT/test_data/image1.jpg"
echo "Then: Describe this image"
echo
exec ./main
