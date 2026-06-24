#!/bin/bash
set -e

# 1. Read SOC information (strip "(null)" / whitespace → default s100)
SOC_RAW=$(cat /sys/class/boardinfo/soc_name 2>/dev/null | tr 'A-Z' 'a-z' | tr -d '()' | xargs)
SOC="${SOC_RAW:-s100}"

# Map SOC to the corresponding pre-quantized PP-OCRv6 model variant.
# Only S100 and S600 builds are published; everything else (S100P, unknown)
# falls back to the S100 build.
case "$SOC" in
  s600) MODEL_SOC="s600" ;;
  *)    MODEL_SOC="s100" ;;
esac

echo "SOC           : $SOC"
echo "Model variant : rdk_${MODEL_SOC}"

MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_${MODEL_SOC}/paddle_ocr"

# 2. Environment Setup
#
# Probe each dependency by what we actually need (a header / pkg-config entry),
# not by package name. This survives Ubuntu version skews — e.g. on the S600
# noble image `libfreetype6-dev` collides with the newer `libfreetype-dev`
# (2.13.x vs 2.11.x). When the header is already in place we don't touch apt.
have_header() {
  # Look in standard include roots and the multiarch dirs.
  local h="$1"
  for d in /usr/include /usr/local/include /usr/include/$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null); do
    [[ -n "$d" && -e "$d/$h" ]] && return 0
  done
  return 1
}

have_pkgconfig() { pkg-config --exists "$1" >/dev/null 2>&1; }

# Probe → apt package candidates. The first candidate that installs cleanly
# wins; if all fail and the probe still misses, we leave the user to install
# manually rather than wedging the system.
declare -A PROBES=(
  [gflags]="header:gflags/gflags.h"
  [polyclipping]="header:polyclipping/clipper.hpp"
  [freetype]="pkgconfig:freetype2"
)
declare -A APT_CANDIDATES=(
  [gflags]="libgflags-dev"
  [polyclipping]="libpolyclipping-dev"
  # libfreetype-dev is the new umbrella package on Ubuntu 24.04 (noble);
  # libfreetype6-dev stays around on 22.04 (jammy) and earlier S100 images.
  [freetype]="libfreetype-dev libfreetype6-dev"
)

probe_one() {
  local key="$1"
  local kind="${PROBES[$key]%%:*}"
  local what="${PROBES[$key]#*:}"
  case "$kind" in
    header)    have_header   "$what" ;;
    pkgconfig) have_pkgconfig "$what" ;;
  esac
}

need_install=()
for key in "${!PROBES[@]}"; do
  if probe_one "$key"; then
    echo "${key} dev files already present, skip"
  else
    echo "${key} dev files missing, will try to install"
    need_install+=("$key")
  fi
done

if (( ${#need_install[@]} > 0 )); then
  echo "Running apt update"
  sudo apt update
  for key in "${need_install[@]}"; do
    installed=false
    for pkg in ${APT_CANDIDATES[$key]}; do
      echo "Trying to install $pkg for $key"
      if sudo apt install -y "$pkg"; then
        installed=true
        break
      fi
      echo "  → $pkg not installable on this distro, trying next candidate"
    done
    if ! $installed; then
      echo "ERROR: failed to satisfy '$key'. Tried: ${APT_CANDIDATES[$key]}" >&2
      echo "       Install a development package providing it manually, then re-run." >&2
      exit 1
    fi
  done
fi

# 3. Model Download (PP-OCRv6, SOC-aware)
DET_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
DET_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"

REC_MODEL_PATH="/opt/hobot/model/${MODEL_SOC}/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
REC_MODEL_URL="${MODEL_BASE_URL}/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"

# Use whichever download tool is available
if command -v wget &>/dev/null; then
  DL_CMD="wget -q -O"
elif command -v curl &>/dev/null; then
  DL_CMD="curl -fL -o"
else
  echo "ERROR: neither wget nor curl found" >&2
  exit 1
fi

echo "Det model  : $DET_MODEL_PATH"
if [[ ! -f "$DET_MODEL_PATH" ]]; then
  echo "Detection model not found, downloading..."
  mkdir -p "$(dirname "$DET_MODEL_PATH")"
  $DL_CMD "$DET_MODEL_PATH" "$DET_MODEL_URL"
  echo "Detection model downloaded successfully"
else
  echo "Detection model already exists, skip download"
fi

echo "Rec model  : $REC_MODEL_PATH"
if [[ ! -f "$REC_MODEL_PATH" ]]; then
  echo "Recognition model not found, downloading..."
  mkdir -p "$(dirname "$REC_MODEL_PATH")"
  $DL_CMD "$REC_MODEL_PATH" "$REC_MODEL_URL"
  echo "Recognition model downloaded successfully"
else
  echo "Recognition model already exists, skip download"
fi

# 4. Model Compilation
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 5. Quick Run
./paddle_ocr \
    --det_model_path "$DET_MODEL_PATH" \
    --rec_model_path "$REC_MODEL_PATH" \
    --test_image ../../../test_data/gt_2322.jpg \
    --label_file ../../../test_data/ppocrv6_dict.txt \
    --font_path ../../../test_data/FangSong.ttf \
    --img_save_path result.jpg
