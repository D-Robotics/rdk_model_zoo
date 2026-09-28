#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${GEMMA4_BUILD_DIR:-$SCRIPT_DIR/build}"
: "${GEMMA4_TARGET:?Set GEMMA4_TARGET=s100/s100p/s600, or use run.sh --target TARGET --build}"
export PATH="$HOME/.cargo/bin:$PATH"
for tool in cmake rustc cargo; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; see runtime/cpp/README.md" >&2; exit 2; }
done
if [[ ! -f "$SCRIPT_DIR/../../third_party/tokenizers-cpp/CMakeLists.txt" ]]; then
  echo "Prepare third_party/install_tokenizers_cpp.sh explicitly before building" >&2
  exit 2
fi
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

CMAKE_ARGS=("-DCARGO_EXECUTABLE=$(command -v cargo)" "-DGEMMA4_TARGET=$GEMMA4_TARGET")
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
