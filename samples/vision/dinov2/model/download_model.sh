#!/usr/bin/env bash
set -e

# Download the pre-quantized DINOv2 ViT-S/14 .hbm model for one S-series march.
# March is auto-detected when omitted; pass nash-e|nash-m|nash-p as the first
# arg to override. An optional second arg restricts the download to one
# variant; omit to download all released variants for that march.
# Model server layout: rdk_model_zoo/rdk_s100/dinov2/{nash-e,nash-m}/ and
# rdk_model_zoo/rdk_s600/dinov2/nash-p/ (S100P models live under rdk_s100).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
case "${march}" in
  nash-e) suffix=nashe; soc_dir=rdk_s100 ;;
  nash-m) suffix=nashm; soc_dir=rdk_s100 ;;
  nash-p) suffix=nashp; soc_dir=rdk_s600 ;;
  *)
    echo "[Error] Unsupported march '${march}'. Expected one of: nash-e|nash-m|nash-p." >&2
    exit 2
    ;;
esac
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/${soc_dir}/dinov2"
variants="${2:-vits14_224_int16}"

download_tmp=""
cleanup_download() {
  if [ -n "${download_tmp}" ]; then
    rm -f "${download_tmp}"
  fi
}
trap cleanup_download EXIT INT TERM

mkdir -p "${SCRIPT_DIR}/${march}"
for v in ${variants}; do
  name="dinov2_${v}_${suffix}.hbm"
  dest="${SCRIPT_DIR}/${march}/${name}"
  if [ ! -s "${dest}" ]; then
    echo "[Info] Downloading ${name} from ${BASE_URL}/${march}/${name}"
    rm -f "${dest}"
    download_tmp="$(mktemp "${dest}.download.XXXXXX")"
    if ! wget -O "${download_tmp}" "${BASE_URL}/${march}/${name}"; then
      echo "[Error] Download failed for ${name}" >&2
      exit 1
    fi
    if [ ! -s "${download_tmp}" ]; then
      echo "[Error] Download produced an empty file for ${name}" >&2
      exit 1
    fi
    mv -f "${download_tmp}" "${dest}"
    download_tmp=""
  fi
done
echo "[Info] DINOv2 ${march} models ready in ${SCRIPT_DIR}/${march}"
