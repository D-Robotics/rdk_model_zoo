#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TARGET_SOC="${TARGET_SOC:-s100p}"
TARGET_SOC="${TARGET_SOC,,}"
case "$TARGET_SOC" in
  s100)
    DEFAULT_MARCH="nash-e"
    DEFAULT_DECODE_CORE_NUM=1
    DEFAULT_HBDK_JOBS=29
    DEFAULT_HBDK_OPT=0
    ;;
  s100p)
    DEFAULT_MARCH="nash-m"
    DEFAULT_DECODE_CORE_NUM=1
    DEFAULT_HBDK_JOBS=29
    DEFAULT_HBDK_OPT=0
    ;;
  s600)
    DEFAULT_MARCH="nash-p"
    DEFAULT_DECODE_CORE_NUM=2
    DEFAULT_HBDK_JOBS=22
    DEFAULT_HBDK_OPT=1
    ;;
  *)
    echo "Unsupported TARGET_SOC=$TARGET_SOC (expected s100, s100p, or s600)" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_text_$TARGET_SOC}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CACHE_LEN="${CACHE_LEN:-4096}"
MEM_LIMIT_GIB="${MEM_LIMIT_GIB:-110}"
MEM_LIMIT_BYTES=$((MEM_LIMIT_GIB * 1024 * 1024 * 1024))

export HBDK_MARCH="${HBDK_MARCH:-$DEFAULT_MARCH}"
export DECODE_CORE_NUM="${DECODE_CORE_NUM:-$DEFAULT_DECODE_CORE_NUM}"
export HBDK_JOBS="${HBDK_JOBS:-$DEFAULT_HBDK_JOBS}"
export HBDK_OPT="${HBDK_OPT:-$DEFAULT_HBDK_OPT}"
export HBDK_CACHE_MODE="${HBDK_CACHE_MODE:-enable}"
export GEMMA4_MAX_L2M_SIZE="${GEMMA4_MAX_L2M_SIZE:-25165824}"
export CHUNK_SIZE CACHE_LEN
export COMPILE_OUTPUT_DIR="$OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/text_decode_resume_$(date +%Y%m%d_%H%M%S).log"

echo "=== Gemma4 Text decode resume ==="
echo "  target_soc:    $TARGET_SOC"
echo "  march:         $HBDK_MARCH"
echo "  decode cores:  $DECODE_CORE_NUM"
echo "  output:        $OUTPUT_DIR"
echo "  chunk/cache:   $CHUNK_SIZE / $CACHE_LEN"
echo "  jobs/opt:      $HBDK_JOBS / $HBDK_OPT"
echo "  memory limit:  ${MEM_LIMIT_GIB} GiB"
echo

free -h
if swapon --show 2>/dev/null | grep -q .; then
  echo "swap: enabled"
else
  echo "WARNING: no swap active. Run: bash $SCRIPT_DIR/setup_swap.sh" >&2
fi
echo

if command -v prlimit >/dev/null 2>&1; then
  echo "Starting with prlimit --as=${MEM_LIMIT_GIB}GiB ..."
  prlimit --as="$MEM_LIMIT_BYTES" \
    python3 -u "$SCRIPT_DIR/compile_text_decode.py" \
    2>&1 | tee "$LOG"
else
  echo "prlimit not found, falling back to ulimit -v ..."
  ULIMIT_KB=$((MEM_LIMIT_GIB * 1024 * 1024))
  ulimit -v "$ULIMIT_KB"
  python3 -u "$SCRIPT_DIR/compile_text_decode.py" \
    2>&1 | tee "$LOG"
fi
