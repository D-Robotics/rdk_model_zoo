#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"
BUILD_DIR="$SCRIPT_DIR/build"
export PATH="$HOME/.cargo/bin:$PATH"

# Usage:
#   ./run.sh                              # interactive main
#   ./run.sh --max_tokens=512             # main with gflags
#   ./run.sh server --port=8000           # OpenAI-compatible HTTP API
#   ./run.sh demo text --prompt "Hello"  # single-shot diagnostic

# Optional proxy for git/cargo/cmake downloads (export before running):
#   export HTTP_PROXY=http://proxy.example.com:8080 HTTPS_PROXY=$HTTP_PROXY
if [[ -n "${HTTP_PROXY:-}" ]]; then
  export http_proxy="${HTTP_PROXY}"
  export https_proxy="${HTTPS_PROXY:-${HTTP_PROXY}}"
fi

SOC=""
if [[ -r /sys/class/boardinfo/soc_name ]]; then
  SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
  echo "SOC         : $SOC"
fi

if [[ "$SOC" == "s600" ]]; then
  unset LD_LIBRARY_PATH GEMMA4_USE_DNN_V3
  export HB_DNN_USER_DEFINED_L2M_SIZES="${HB_DNN_USER_DEFINED_L2M_SIZES:-6:6:6:6}"
  echo "DNN mode    : system runtime (V2)"
  echo "L2M sizes   : $HB_DNN_USER_DEFINED_L2M_SIZES"
fi

PKGS=(cmake g++ libopencv-dev libgflags-dev nlohmann-json3-dev cargo wget git curl)
need_update=false
for pkg in "${PKGS[@]}"; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    need_update=true
    break
  fi
done

if $need_update; then
  if command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt)
  elif [[ ${EUID:-$(id -u)} -eq 0 ]]; then
    APT=(apt)
  else
    echo "ERROR: missing packages require root privileges or sudo." >&2
    exit 1
  fi
  echo "Running apt update (packages missing)..."
  "${APT[@]}" update
fi

for pkg in "${PKGS[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "$pkg already installed"
  else
    echo "Installing $pkg..."
    "${APT[@]}" install -y "$pkg"
  fi
done

echo "GEMMA4_HOME : $GEMMA4_HOME"
GEMMA4_HOME="$GEMMA4_HOME" bash "$SAMPLE_ROOT/model/download_model.sh"
bash "$SAMPLE_ROOT/third_party/install_tokenizers_cpp.sh"

mkdir -p "$BUILD_DIR"
RUST_TOOLCHAIN_ID="$(rustc --version); $(command -v cargo) $(cargo --version)"
RUST_VERSION_FILE="$BUILD_DIR/.tokenizers-rust-version"
PREVIOUS_RUST_TOOLCHAIN_ID="$(cat "$RUST_VERSION_FILE" 2>/dev/null || true)"
if [[ -d "$BUILD_DIR/tokenizers-cpp/release" &&
      ! -f "$BUILD_DIR/tokenizers-cpp/release/libtokenizers_c.a" &&
      "$PREVIOUS_RUST_TOOLCHAIN_ID" != "$RUST_TOOLCHAIN_ID" ]]; then
  echo "Rust toolchain changed; cleaning stale tokenizers build artifacts."
  rm -rf "$BUILD_DIR/tokenizers-cpp/release"
fi
printf '%s\n' "$RUST_TOOLCHAIN_ID" > "$RUST_VERSION_FILE"

# tokenizers-cpp sets CARGO_TARGET=aarch64-unknown-linux-gnu on aarch64 even
# for native builds, so cc-rs looks for aarch64-unknown-linux-gnu-gcc. Stock
# Ubuntu provides aarch64-linux-gnu-gcc (no "unknown"). Create aliases under
# build/ so a non-root user can compile without modifying /usr/local/bin.
mkdir -p "$BUILD_DIR"
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]]; then
  TOOL_ALIAS_DIR="$BUILD_DIR/toolchain_aliases"
  mkdir -p "$TOOL_ALIAS_DIR"
  for tool in gcc g++ ar; do
    if ! command -v "aarch64-unknown-linux-gnu-$tool" >/dev/null 2>&1; then
      ln -sf "$(command -v "aarch64-linux-gnu-$tool" || command -v "$tool")" \
             "$TOOL_ALIAS_DIR/aarch64-unknown-linux-gnu-$tool"
    fi
  done
  export PATH="$TOOL_ALIAS_DIR:$PATH"
fi

CMAKE_ARGS=("-DCARGO_EXECUTABLE=$(command -v cargo)")
if [[ -n "${GEMMA4_ABSL_PREFIX:-}" ]]; then
  ABSL_PREFIX="${GEMMA4_ABSL_PREFIX%/}"
  CMAKE_ARGS+=(
    -DSPM_ABSL_PROVIDER=package
    "-DCMAKE_PREFIX_PATH=$ABSL_PREFIX"
    "-Dabsl_DIR=$ABSL_PREFIX/lib/cmake/absl"
  )
fi
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --parallel "$(nproc)"

export GEMMA4_HOME
APP="${GEMMA4_APP:-main}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  APP="$1"
  shift
fi
case "$APP" in
  main)
    BINARY=main
    ;;
  server|gemma4_server)
    BINARY=gemma4_server
    ;;
  demo|gemma4_demo)
    BINARY=gemma4_demo
    ;;
  text_bench|gemma4_text_bench)
    BINARY=gemma4_text_bench
    ;;
  golden_verify|gemma4_golden_verify)
    BINARY=gemma4_golden_verify
    ;;
  *)
    echo "ERROR: unknown app $APP" >&2
    echo "Expected: main, server, demo, text_bench, or golden_verify" >&2
    exit 2
    ;;
esac

echo
echo "Starting $BINARY (Ctrl+C to exit)..."
if [[ "$BINARY" == "main" ]]; then
  echo "Try: /image $SAMPLE_ROOT/test_data/image1.jpg"
  echo "Then: Describe this image"
elif [[ "$BINARY" == "gemma4_server" ]]; then
  echo "OpenAI base URL: http://<board-ip>:8000/v1"
fi
echo
exec "$BUILD_DIR/$BINARY" "$@"
