#!/usr/bin/env bash
set -e

# RDK-S YOLO26 Depth launcher. Default variant is `n`; pass a different variant as
# the first arg. The march is auto-detected so only the matching .hbm is
# downloaded, then main.py re-detects the board to resolve the exact model path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_DIR="${SAMPLE_DIR}/model"
VARIANT="${1:-n}"

soc="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/soc_name 2>/dev/null || true)"
btype="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/board_type 2>/dev/null || true)"
case "${btype}:${soc}" in
  *s100p*|*:s100p) march="nash-m" ;;
  *:s100) march="nash-e" ;;
  *:s600) march="nash-p" ;;
  *) march="nash-e" ;;
esac
case "${march}" in nash-m) suffix=nashm;; nash-p) suffix=nashp;; *) suffix=nashe;; esac
case "${VARIANT}" in
  l|x) MODEL="${MODEL_DIR}/${march}/yolo26${VARIANT}_depth_lite_${suffix}_768x768.hbm" ;;
  *)   MODEL="${MODEL_DIR}/${march}/yolo26${VARIANT}_depth_${suffix}_768x768_nv12.hbm" ;;
esac

if [ ! -f "${MODEL}" ]; then
  bash "${MODEL_DIR}/download_model.sh" "${march}" "${VARIANT}"
fi

cd "${SCRIPT_DIR}"
python3 main.py --variant "${VARIANT}" --input "${SAMPLE_DIR}/test_data/bus.jpg" --output "${SCRIPT_DIR}/output"
