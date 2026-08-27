#!/usr/bin/env bash
set -e

# Download the pre-quantized YOLO26 Depth .hbm models for one S-series march.
# March is auto-detected when omitted; pass nash-e|nash-m|nash-p as the first
# arg. An optional second arg restricts the download to one variant (n|s|m|l|x);
# omit to download all five variants for that march.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s/yolo26_depth"

march="${1:-}"
if [ -z "${march}" ]; then
  soc="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/soc_name 2>/dev/null || true)"
  btype="$(tr '[:upper:]' '[:lower:]' < /sys/class/boardinfo/board_type 2>/dev/null || true)"
  case "${btype}:${soc}" in
    *s100p*|*:s100p) march="nash-m" ;;
    *:s100) march="nash-e" ;;
    *:s600) march="nash-p" ;;
    *) march="nash-e" ;;
  esac
fi
case "${march}" in nash-m) suffix=nashm;; nash-p) suffix=nashp;; *) suffix=nashe;; esac
variants="${2:-n s m l x}"

mkdir -p "${SCRIPT_DIR}/${march}"
for v in ${variants}; do
  case "${v}" in
    l|x) name="yolo26${v}_depth_lite_${suffix}_768x768.hbm" ;;
    *)   name="yolo26${v}_depth_${suffix}_768x768_nv12.hbm" ;;
  esac
  dest="${SCRIPT_DIR}/${march}/${name}"
  if [ ! -f "${dest}" ]; then
    echo "[Info] Downloading ${name} from ${BASE_URL}/${march}/${name}"
    wget -O "${dest}" "${BASE_URL}/${march}/${name}"
  fi
done
echo "[Info] YOLO26 Depth ${march} models ready in ${SCRIPT_DIR}/${march}"
