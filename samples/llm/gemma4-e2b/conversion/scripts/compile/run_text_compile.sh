#!/bin/bash
# Text HBM 编译脚本（prefill + decode → link 成单个 HBM）
# 用法: bash run_text_compile.sh
# 环境变量（可选）:
#   GEMMA4_E2B_DIR   - HuggingFace 权重目录
#   OUTPUT_DIR       - 编译输出目录
#   CALIB_TEXT_DIR   - 校准文本目录
#   CHUNK_SIZE       - prefill chunk 大小 (默认 256)
#   CACHE_LEN        - KV cache 长度 (默认 4096)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GEMMA4_E2B_DIR="${GEMMA4_E2B_DIR:-$REPO_ROOT/gemma4-e2b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_text}"
CALIB_TEXT_DIR="${CALIB_TEXT_DIR:-$REPO_ROOT/calibration_data/text}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
CACHE_LEN="${CACHE_LEN:-4096}"

OELLM_BUILD=$(python -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'oellm_build.py'))")

mkdir -p "$OUTPUT_DIR"
ulimit -s unlimited

echo "=== Text compile ==="
echo "  model:     $GEMMA4_E2B_DIR"
echo "  output:    $OUTPUT_DIR"
echo "  calib:     $CALIB_TEXT_DIR"
echo "  chunk:     $CHUNK_SIZE"
echo "  cache_len: $CACHE_LEN"
echo

python -u "$OELLM_BUILD" \
    --model_name gemma4-e2b-text \
    --march nash-m \
    --input_model_path "$GEMMA4_E2B_DIR" \
    --output_model_path "$OUTPUT_DIR" \
    --calib_text_path "$CALIB_TEXT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --cache_len "$CACHE_LEN" \
    --device cpu \
    --core_num 1 \
    2>&1 | tee "$OUTPUT_DIR/text_compile.log"

echo "Done: $OUTPUT_DIR/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
