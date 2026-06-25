#!/usr/bin/env bash
# Remote HBM verify — run on a PC that can SSH to the S100P board.
# HBM mode cannot run locally; it must execute on the board via SSH.
#
# Usage: bash run_remote_hbm_verify.sh
# Env vars: BOARD_IP, BOARD_USER, BOARD_PORT, SDK_ROOT, GEMMA_ROOT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GEMMA_ROOT="${GEMMA_ROOT:-$REPO_ROOT}"
MODEL_DIR="${MODEL_DIR:-$GEMMA_ROOT/gemma4-e2b}"
VISION_HBM="${VISION_HBM:-$GEMMA_ROOT/output/gemma4_e2b_vision/gemma4-e2b_vit_ptq.hbm}"
TEXT_HBM="${TEXT_HBM:-$GEMMA_ROOT/output/gemma4_e2b_text/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm}"
CALIB_IMG="${CALIB_IMG:-$GEMMA_ROOT/calibration_data/images/coco_00_000000000802.jpg}"
CALIB_TEXT="${CALIB_TEXT:-$GEMMA_ROOT/calibration_data/text_verify}"

BOARD_IP="${BOARD_IP:-192.168.127.10}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
REMOTE_PATH="${REMOTE_PATH:-/tmp/}"

VERIFIER=$(python -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'verifier_cli.py'))")

echo "=== Preflight ==="
python -c "import leap_llm; print('leap_llm ok')"
ssh -p "$BOARD_PORT" -o ConnectTimeout=10 "${BOARD_USER}@${BOARD_IP}" \
  "df -h /tmp && free -h"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$GEMMA_ROOT/output/remote_verify_${TS}"
mkdir -p "$LOG_DIR"

echo "=== Vision remote ==="
python "$VERIFIER" \
  --model_name gemma4-e2b-vision \
  --model_dir "$MODEL_DIR" \
  --hbm_vlm_model_path "$VISION_HBM" \
  --input_image_path "$CALIB_IMG" \
  --remote_ip "$BOARD_IP" \
  --remote_path "$REMOTE_PATH" \
  --username "$BOARD_USER" \
  --port "$BOARD_PORT" \
  2>&1 | tee "$LOG_DIR/vision_remote.log"

echo "=== Text remote ==="
python "$VERIFIER" \
  --model_name gemma4-e2b-text \
  --model_dir "$MODEL_DIR" \
  --hbm_llm_model_path "$TEXT_HBM" \
  --input_text_path "$CALIB_TEXT" \
  --chunk_size 256 \
  --cache_len 4096 \
  --remote_ip "$BOARD_IP" \
  --remote_path "$REMOTE_PATH" \
  --username "$BOARD_USER" \
  --port "$BOARD_PORT" \
  2>&1 | tee "$LOG_DIR/text_remote.log"

echo "=== Done. Logs: $LOG_DIR ==="
