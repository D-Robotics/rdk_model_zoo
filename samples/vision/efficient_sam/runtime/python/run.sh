#!/usr/bin/env bash
set -e

# RDK-S runs on three boards (S100/S100P/S600 -> nash-e/nash-m/nash-p). The march
# is detected here so we only download the matching .hbm pair, then main.py
# re-detects the board in Python to resolve the exact model path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SAMPLE_DIR}/model"

soc="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/soc_name 2>/dev/null || true)"
btype="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/board_type 2>/dev/null || true)"
case "${btype}:${soc}" in
  *s100p*|*:s100p) march="nash-m" ;;
  *:s100) march="nash-e" ;;
  *:s600) march="nash-p" ;;
  *) march="nash-e" ;;
esac
case "${march}" in nash-m) suffix=nashm;; nash-p) suffix=nashp;; *) suffix=nashe;; esac
ENCODER="${MODEL_DIR}/${march}/efficient_sam_vitt_encoder_512x512_${suffix}.hbm"
DECODER="${MODEL_DIR}/${march}/efficient_sam_vitt_decoder_512_${suffix}.hbm"

if [ ! -f "${ENCODER}" ] || [ ! -f "${DECODER}" ]; then
  bash "${MODEL_DIR}/download_model.sh" "${march}"
fi

cd "${SCRIPT_DIR}"
python3 main.py "$@"
