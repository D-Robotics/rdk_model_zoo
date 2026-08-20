#!/usr/bin/env bash
set -e

# Download the pre-quantized MobileSAM .hbm pair for one S-series march.
# March is auto-detected when omitted; pass nash-e|nash-m|nash-p to override.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# BASE_URL is derived below from the march's platform (rdk_s100 / rdk_s600).

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
case "${march}" in nash-m) suffix=nashm; platform=s100;; nash-p) suffix=nashp; platform=s600;; *) suffix=nashe; platform=s100;; esac
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${platform}/mobile_sam"

mkdir -p "${SCRIPT_DIR}/${march}"
for name in \
  "mobile_sam_image_encoder_norm_512x512_${suffix}.hbm" \
  "mobile_sam_decoder_512_${suffix}.hbm"; do
  dest="${SCRIPT_DIR}/${march}/${name}"
  if [ ! -f "${dest}" ]; then
    echo "[Info] Downloading ${name} from ${BASE_URL}/${march}/${name}"
    wget -O "${dest}" "${BASE_URL}/${march}/${name}"
  fi
done
echo "[Info] MobileSAM ${march} models ready in ${SCRIPT_DIR}/${march}"
