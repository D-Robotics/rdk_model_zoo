#!/bin/bash
# Text decode 续编脚本（decode 阶段 OOM 后可单独续编）
# 用于 prefill 已编译完成、decode 需要重试或续编的场景
# 用法: bash run_text_decode_resume.sh
# 环境变量（可选）:
#   OUTPUT_DIR       - 编译输出目录
#   MEM_LIMIT_GIB    - 虚拟内存上限 GiB (默认 110)
#   HBDK_JOBS        - 编译并行度 (默认 29)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_text}"
MEM_LIMIT_GIB="${MEM_LIMIT_GIB:-110}"
MEM_LIMIT_BYTES=$((MEM_LIMIT_GIB * 1024 * 1024 * 1024))

export HBDK_JOBS="${HBDK_JOBS:-29}"
export HBDK_OPT="${HBDK_OPT:-0}"
export HBDK_CACHE_MODE="${HBDK_CACHE_MODE:-enable}"

LOG="${OUTPUT_DIR}/text_decode_resume.log"

echo "=== Text decode resume ==="
echo "  HBDK_JOBS=$HBDK_JOBS  HBDK_OPT=$HBDK_OPT  MEM_LIMIT=${MEM_LIMIT_GIB}GiB"
echo

echo "=== memory / swap ==="
free -h
if swapon --show 2>/dev/null | grep -q .; then
  echo "swap: enabled"
else
  echo "WARNING: no swap active. Run first: bash $SCRIPT_DIR/setup_swap.sh"
fi
echo

# 设置 OUTPUT_DIR 环境变量给 Python 脚本
export COMPILE_OUTPUT_DIR="$OUTPUT_DIR"

if command -v prlimit >/dev/null 2>&1; then
  echo "Starting with prlimit --as=${MEM_LIMIT_GIB}GiB ..."
  exec prlimit --as="$MEM_LIMIT_BYTES" \
    python -u "$SCRIPT_DIR/compile_text_decode.py" \
    2>&1 | tee "$LOG"
else
  echo "prlimit not found, falling back to ulimit -v ..."
  ULIMIT_KB=$((MEM_LIMIT_GIB * 1024 * 1024))
  ulimit -v "$ULIMIT_KB"
  exec python -u "$SCRIPT_DIR/compile_text_decode.py" \
    2>&1 | tee "$LOG"
fi
