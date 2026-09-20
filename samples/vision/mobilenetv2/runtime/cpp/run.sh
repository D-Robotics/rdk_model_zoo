#!/usr/bin/env bash
set -euo pipefail

# Build and run only.  Toolchain/runtime dependencies and model preparation
# are explicit README steps; this launcher never invokes apt, pip, or wget.
# S-series C++ flow only (the X5 source delivered Python alone).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
SAMPLE_DIR="$SCRIPT_DIR/../.."

# Board identity is read once from the same source the Python flow uses.
# s100p and unknown boards are rejected instead of silently falling back to
# the s100 artifact (the legacy source launcher did fall back).
SOC_RAW="$(tr -d '() \t\r\n' < /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z')"
case "$SOC_RAW" in
  s100|s600) ;;
  *)
    echo "error: unrecognized or unsupported SoC '${SOC_RAW:-unknown}';" \
         "this C++ flow supports s100/s600 only" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-$SAMPLE_DIR/model/$SOC_RAW/mobilenetv2_224x224_nv12.hbm}"
TEST_IMAGE="${TEST_IMAGE:-$SAMPLE_DIR/test_data/zebra_cls.jpg}"
LABEL_FILE="${LABEL_FILE:-$SAMPLE_DIR/test_data/imagenet1000_labels.txt}"

has_option() {
  local name="$1"
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$name" || "$value" == "$name="* ]] && return 0
  done
  return 1
}

option_value() {
  local name="$1"
  shift
  local previous=""
  local value
  for value in "$@"; do
    if [[ "$previous" == "$name" ]]; then
      printf '%s' "$value"
      return 0
    fi
    if [[ "$value" == "$name="* ]]; then
      printf '%s' "${value#*=}"
      return 0
    fi
    previous="$value"
  done
  return 1
}

# A caller may override all three file flags. Resolve those values before the
# existence checks so a prepared model outside the sample directory works.
if value="$(option_value --model_path "$@")"; then MODEL_PATH="$value"; fi
if value="$(option_value --test_img "$@")"; then TEST_IMAGE="$value"; fi
if value="$(option_value --label_file "$@")"; then LABEL_FILE="$value"; fi
TOP_K="${TOP_K:-5}"
if value="$(option_value --top_k "$@")"; then TOP_K="$value"; fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Prepare it explicitly with samples/vision/mobilenetv2/model/download.sh $SOC_RAW." >&2
  exit 2
fi
if [[ ! -f "$TEST_IMAGE" ]]; then
  echo "Test image not found: $TEST_IMAGE" >&2
  exit 2
fi
if [[ ! -f "$LABEL_FILE" ]]; then
  echo "Label file not found: $LABEL_FILE" >&2
  exit 2
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-$(nproc)}"

NATIVE_ARGS=("$@")
if ! has_option --model_path "${NATIVE_ARGS[@]}"; then
  NATIVE_ARGS+=(--model_path "$MODEL_PATH")
fi
if ! has_option --test_img "${NATIVE_ARGS[@]}"; then
  NATIVE_ARGS+=(--test_img "$TEST_IMAGE")
fi
if ! has_option --label_file "${NATIVE_ARGS[@]}"; then
  NATIVE_ARGS+=(--label_file "$LABEL_FILE")
fi
if ! has_option --top_k "${NATIVE_ARGS[@]}"; then
  NATIVE_ARGS+=(--top_k "$TOP_K")
fi
exec "$BUILD_DIR/mobilenetv2" "${NATIVE_ARGS[@]}"
