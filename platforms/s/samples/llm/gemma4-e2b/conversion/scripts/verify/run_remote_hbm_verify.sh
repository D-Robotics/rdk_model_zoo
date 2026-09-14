#!/usr/bin/env bash
# Run HBM verification on an RDK S-series board reachable over SSH.
#
# Usage:
#   BOARD_IP=192.0.2.10 TARGET_SOC=s600 bash run_remote_hbm_verify.sh
#
# Optional env vars: BOARD_USER, BOARD_PORT, REMOTE_PATH, PYTHON_BIN,
# MODEL_DIR, VISION_HBM, TEXT_HBM, CALIB_IMG, CALIB_TEXT, CONVERSION_ROOT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERSION_ROOT="${CONVERSION_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

TARGET_SOC="${TARGET_SOC:-s100p}"
TARGET_SOC="${TARGET_SOC,,}"
case "$TARGET_SOC" in
  s100|s100p|s600)
    ;;
  *)
    echo "ERROR: unsupported TARGET_SOC=$TARGET_SOC" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-$CONVERSION_ROOT/gemma4-e2b}"
VISION_HBM="${VISION_HBM:-$CONVERSION_ROOT/output/gemma4_e2b_vision_$TARGET_SOC/gemma4-e2b_vit_ptq.hbm}"
TEXT_HBM="${TEXT_HBM:-$CONVERSION_ROOT/output/gemma4_e2b_text_$TARGET_SOC/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm}"
CALIB_IMG="${CALIB_IMG:-$CONVERSION_ROOT/calibration_data/images/coco_00_000000000802.jpg}"
CALIB_TEXT="${CALIB_TEXT:-$CONVERSION_ROOT/calibration_data/text_verify}"

BOARD_IP="${BOARD_IP:-}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PORT="${BOARD_PORT:-22}"
REMOTE_PATH="${REMOTE_PATH:-/tmp/}"

if [[ -z "$BOARD_IP" ]]; then
  echo "ERROR: BOARD_IP is required." >&2
  echo "Example: BOARD_IP=192.0.2.10 TARGET_SOC=$TARGET_SOC bash $0" >&2
  exit 2
fi
for required_path in "$MODEL_DIR" "$VISION_HBM" "$TEXT_HBM" "$CALIB_IMG" "$CALIB_TEXT"; do
  if [[ ! -e "$required_path" ]]; then
    echo "ERROR: required path does not exist: $required_path" >&2
    exit 2
  fi
done
command -v ssh >/dev/null 2>&1 || {
  echo "ERROR: ssh is required" >&2
  exit 2
}

VERIFIER=$("$PYTHON_BIN" -c "import leap_llm, os; print(os.path.join(os.path.dirname(leap_llm.__file__), 'apis', 'verifier_cli.py'))")

echo "=== Preflight ==="
echo "target_soc : $TARGET_SOC"
echo "board      : ${BOARD_USER}@${BOARD_IP}:${BOARD_PORT}"
echo "vision_hbm : $VISION_HBM"
echo "text_hbm   : $TEXT_HBM"
"$PYTHON_BIN" -c "import leap_llm; print('leap_llm ok')"
ssh -p "$BOARD_PORT" -o ConnectTimeout=10 "${BOARD_USER}@${BOARD_IP}" \
  "df -h /tmp && free -h"

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$CONVERSION_ROOT/output/remote_verify_${TARGET_SOC}_${TS}"
mkdir -p "$LOG_DIR"

echo "=== Vision remote ==="
"$PYTHON_BIN" "$VERIFIER" \
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
"$PYTHON_BIN" "$VERIFIER" \
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
