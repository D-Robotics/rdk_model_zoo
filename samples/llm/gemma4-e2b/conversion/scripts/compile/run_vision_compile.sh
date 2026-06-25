#!/bin/bash
# Vision HBM 编译脚本
# 用法: bash run_vision_compile.sh
# 环境变量（可选）:
#   GEMMA4_E2B_DIR   - HuggingFace 权重目录 (默认: ../../gemma4-e2b)
#   OUTPUT_DIR       - 编译输出目录 (默认: ../../output/gemma4_e2b_vision)
#   CALIB_IMAGE_DIR  - 校准图像目录 (默认: ../../calibration_data/images)
#   DEVICE           - 校准设备 (默认: cuda:0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GEMMA4_E2B_DIR="${GEMMA4_E2B_DIR:-$REPO_ROOT/gemma4-e2b}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output/gemma4_e2b_vision}"
CALIB_IMAGE_DIR="${CALIB_IMAGE_DIR:-$REPO_ROOT/calibration_data/images}"
DEVICE="${DEVICE:-cuda:0}"

OELLM_BUILD=$(python -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'oellm_build.py'))")

mkdir -p "$OUTPUT_DIR"

# 增大栈大小，解决 hbdk4 compile_hbo segfault
ulimit -s unlimited
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Vision compile ==="
echo "  model:   $GEMMA4_E2B_DIR"
echo "  output:  $OUTPUT_DIR"
echo "  calib:   $CALIB_IMAGE_DIR"
echo "  device:  $DEVICE"
echo

python -u "$OELLM_BUILD" \
    --model_name gemma4-e2b-vision \
    --march nash-m \
    --input_model_path "$GEMMA4_E2B_DIR" \
    --output_model_path "$OUTPUT_DIR" \
    --calib_image_path "$CALIB_IMAGE_DIR" \
    --device "$DEVICE" \
    --vit_core_num 1 \
    2>&1 | tee "$OUTPUT_DIR/vision_compile.log"

echo "Done: $OUTPUT_DIR/gemma4-e2b_vit_ptq.hbm"
