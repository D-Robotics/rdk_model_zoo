#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_SOC="${TARGET_SOC:-s100p}"
TARGET_SOC="${TARGET_SOC,,}"
case "$TARGET_SOC" in
  s100)
    DEFAULT_MARCH="nash-e"
    DEFAULT_VIT_CORE_NUM=1
    DEFAULT_HBDK_JOBS=25
    DEFAULT_HBDK_OPT=0
    ;;
  s100p)
    DEFAULT_MARCH="nash-m"
    DEFAULT_VIT_CORE_NUM=1
    DEFAULT_HBDK_JOBS=25
    DEFAULT_HBDK_OPT=0
    ;;
  s600)
    DEFAULT_MARCH="nash-p"
    DEFAULT_VIT_CORE_NUM=4
    DEFAULT_HBDK_JOBS=22
    DEFAULT_HBDK_OPT=1
    ;;
  *)
    echo "Unsupported TARGET_SOC=$TARGET_SOC (expected s100, s100p, or s600)" >&2
    exit 2
    ;;
esac

HBDK_MARCH="${HBDK_MARCH:-$DEFAULT_MARCH}"
VIT_CORE_NUM="${VIT_CORE_NUM:-$DEFAULT_VIT_CORE_NUM}"
HBDK_JOBS="${HBDK_JOBS:-$DEFAULT_HBDK_JOBS}"
HBDK_OPT="${HBDK_OPT:-$DEFAULT_HBDK_OPT}"
GEMMA4_MAX_L2M_SIZE="${GEMMA4_MAX_L2M_SIZE:-25165824}"

GEMMA4_E2B_DIR="${GEMMA4_E2B_DIR:-$REPO_ROOT/gemma4-e2b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_vision_$TARGET_SOC}"
CALIB_IMAGE_DIR="${CALIB_IMAGE_DIR:-$REPO_ROOT/calibration_data/images}"
DEVICE="${DEVICE:-cuda:0}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/.hbdk_cache}"

[[ -d "$GEMMA4_E2B_DIR" ]] || {
  echo "Missing Hugging Face model directory: $GEMMA4_E2B_DIR" >&2
  exit 2
}
[[ -d "$CALIB_IMAGE_DIR" ]] || {
  echo "Missing calibration image directory: $CALIB_IMAGE_DIR" >&2
  exit 2
}

python3 "$SCRIPT_DIR/../calibration/download_coco_images.py" \
  --output-dir "$CALIB_IMAGE_DIR" \
  --verify-only

OELLM_BUILD="$(python3 -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'oellm_build.py'))")"

mkdir -p "$OUTPUT_DIR" "$CACHE_PATH"
ulimit -s unlimited
export PYTHONUNBUFFERED=1
export GEMMA4_MAX_L2M_SIZE

LOG="$OUTPUT_DIR/vision_compile_$(date +%Y%m%d_%H%M%S).log"

echo "=== Gemma4 Vision compile ==="
echo "  target_soc:   $TARGET_SOC"
echo "  march:        $HBDK_MARCH"
echo "  BPU cores:    $VIT_CORE_NUM"
echo "  model:        $GEMMA4_E2B_DIR"
echo "  output:       $OUTPUT_DIR"
echo "  calibration:  $CALIB_IMAGE_DIR (verified COCO val2017)"
echo "  device:       $DEVICE"
echo "  jobs/opt:     $HBDK_JOBS / $HBDK_OPT"
echo

python3 -u "$OELLM_BUILD" \
  --model_name gemma4-e2b-vision \
  --march "$HBDK_MARCH" \
  --input_model_path "$GEMMA4_E2B_DIR" \
  --output_model_path "$OUTPUT_DIR" \
  --calib_image_path "$CALIB_IMAGE_DIR" \
  --device "$DEVICE" \
  --vit_core_num "$VIT_CORE_NUM" \
  --jobs "$HBDK_JOBS" \
  --opt "$HBDK_OPT" \
  --cache_path "$CACHE_PATH" \
  2>&1 | tee "$LOG"

HBM="$(find "$OUTPUT_DIR" -maxdepth 2 -name '*_vit_ptq*.hbm' -print -quit)"
[[ -n "$HBM" ]] || {
  echo "No Vision HBM produced; see $LOG" >&2
  exit 1
}
echo "vision_hbm=$HBM"
