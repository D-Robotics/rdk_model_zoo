#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_SOC="${TARGET_SOC:-s100p}"
TARGET_SOC="${TARGET_SOC,,}"
case "$TARGET_SOC" in
  s100)
    DEFAULT_MARCH="nash-e"
    DEFAULT_PREFILL_CORE_NUM=1
    DEFAULT_DECODE_CORE_NUM=1
    DEFAULT_HBDK_JOBS=29
    DEFAULT_HBDK_OPT=0
    ;;
  s100p)
    DEFAULT_MARCH="nash-m"
    DEFAULT_PREFILL_CORE_NUM=1
    DEFAULT_DECODE_CORE_NUM=1
    DEFAULT_HBDK_JOBS=29
    DEFAULT_HBDK_OPT=0
    ;;
  s600)
    DEFAULT_MARCH="nash-p"
    DEFAULT_PREFILL_CORE_NUM=2
    DEFAULT_DECODE_CORE_NUM=2
    DEFAULT_HBDK_JOBS=22
    DEFAULT_HBDK_OPT=1
    ;;
  *)
    echo "Unsupported TARGET_SOC=$TARGET_SOC (expected s100, s100p, or s600)" >&2
    exit 2
    ;;
esac

HBDK_MARCH="${HBDK_MARCH:-$DEFAULT_MARCH}"
PREFILL_CORE_NUM="${PREFILL_CORE_NUM:-$DEFAULT_PREFILL_CORE_NUM}"
DECODE_CORE_NUM="${DECODE_CORE_NUM:-$DEFAULT_DECODE_CORE_NUM}"
HBDK_JOBS="${HBDK_JOBS:-$DEFAULT_HBDK_JOBS}"
HBDK_OPT="${HBDK_OPT:-$DEFAULT_HBDK_OPT}"
GEMMA4_MAX_L2M_SIZE="${GEMMA4_MAX_L2M_SIZE:-25165824}"

GEMMA4_E2B_DIR="${GEMMA4_E2B_DIR:-$REPO_ROOT/gemma4-e2b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_text_$TARGET_SOC}"
CALIB_TEXT_DIR="${CALIB_TEXT_DIR:-$REPO_ROOT/calibration_data/text}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CACHE_LEN="${CACHE_LEN:-4096}"
DEVICE="${DEVICE:-cpu}"
CACHE_PATH="${CACHE_PATH:-$OUTPUT_DIR/.hbdk_cache}"

[[ -d "$GEMMA4_E2B_DIR" ]] || {
  echo "Missing Hugging Face model directory: $GEMMA4_E2B_DIR" >&2
  exit 2
}
[[ -e "$CALIB_TEXT_DIR" ]] || {
  echo "Missing text calibration data: $CALIB_TEXT_DIR" >&2
  exit 2
}
if [[ -d "$CALIB_TEXT_DIR" ]] && ! find "$CALIB_TEXT_DIR" -type f -print -quit | grep -q .; then
  echo "Text calibration directory is empty: $CALIB_TEXT_DIR" >&2
  exit 2
fi

OELLM_BUILD="$(python3 -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'oellm_build.py'))")"

mkdir -p "$OUTPUT_DIR" "$CACHE_PATH"
ulimit -s unlimited
export PYTHONUNBUFFERED=1
export GEMMA4_MAX_L2M_SIZE

LOG="$OUTPUT_DIR/text_compile_$(date +%Y%m%d_%H%M%S).log"

echo "=== Gemma4 Text compile ==="
echo "  target_soc:    $TARGET_SOC"
echo "  march:         $HBDK_MARCH"
echo "  prefill cores: $PREFILL_CORE_NUM"
echo "  decode cores:  $DECODE_CORE_NUM"
echo "  model:         $GEMMA4_E2B_DIR"
echo "  output:        $OUTPUT_DIR"
echo "  calibration:   $CALIB_TEXT_DIR"
echo "  chunk/cache:   $CHUNK_SIZE / $CACHE_LEN"
echo "  jobs/opt:      $HBDK_JOBS / $HBDK_OPT"
echo

python3 -u "$OELLM_BUILD" \
  --model_name gemma4-e2b-text \
  --march "$HBDK_MARCH" \
  --input_model_path "$GEMMA4_E2B_DIR" \
  --output_model_path "$OUTPUT_DIR" \
  --calib_text_path "$CALIB_TEXT_DIR" \
  --chunk_size "$CHUNK_SIZE" \
  --cache_len "$CACHE_LEN" \
  --device "$DEVICE" \
  --prefill_core_num "$PREFILL_CORE_NUM" \
  --decode_core_num "$DECODE_CORE_NUM" \
  --jobs "$HBDK_JOBS" \
  --opt "$HBDK_OPT" \
  --cache_path "$CACHE_PATH" \
  2>&1 | tee "$LOG"

HBM="$(find "$OUTPUT_DIR" -maxdepth 2 -name '*_ptq*.hbm' -not -name '*_vit_ptq*.hbm' -print -quit)"
TOK="$(find "$OUTPUT_DIR" -maxdepth 2 -name 'tok_embeddings.bin' -print -quit)"
[[ -n "$HBM" && -n "$TOK" ]] || {
  echo "Missing Text artifacts; see $LOG" >&2
  exit 1
}
echo "text_hbm=$HBM"
echo "tok_embeddings=$TOK"
