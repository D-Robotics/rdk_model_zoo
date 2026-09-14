#!/bin/bash
set -euo pipefail

GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"
SOC="${GEMMA4_SOC:-}"
if [[ -z "$SOC" && -r /sys/class/boardinfo/soc_name ]]; then
  SOC=$(tr 'A-Z' 'a-z' </sys/class/boardinfo/soc_name)
fi
SOC="${SOC:-s100p}"

S100P_MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/gemma4_e2b/model"
S600_MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/gemma4_e2b/model"
COMMON_MODEL_BASE_URL="${GEMMA4_COMMON_MODEL_BASE_URL:-$S100P_MODEL_BASE_URL}"
TOKENIZER_BASE_URL="${GEMMA4_TOKENIZER_BASE_URL:-https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/gemma4_e2b/tokenizer}"
MODEL_BASE_URL="${GEMMA4_MODEL_BASE_URL:-}"

case "$SOC" in
  s100p)
    MODEL_BASE_URL="${MODEL_BASE_URL:-$S100P_MODEL_BASE_URL}"
    ;;
  s100)
    ;;
  s600)
    MODEL_BASE_URL="${MODEL_BASE_URL:-$S600_MODEL_BASE_URL}"
    ;;
  *)
    echo "ERROR: unsupported SoC '$SOC'. Set GEMMA4_SOC and GEMMA4_MODEL_BASE_URL."
    exit 1
    ;;
esac

download_file() {
  local base_url="$1"
  local file_name="$2"
  local output_dir="$3"
  local destination="$output_dir/$file_name"
  if [[ -s "$destination" ]]; then
    echo "Found $destination"
    return
  fi
  mkdir -p "$output_dir"
  echo "Downloading $file_name..."
  wget -c -O "$destination.part" "$base_url/$file_name"
  mv -f "$destination.part" "$destination"
}

echo "GEMMA4_HOME : $GEMMA4_HOME"
echo "SOC         : $SOC"

hbm_files=(
  gemma4-e2b_vit_ptq.hbm
  gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm
)
missing_hbm=false
for model_file in "${hbm_files[@]}"; do
  if [[ ! -s "$GEMMA4_HOME/model/$model_file" ]]; then
    missing_hbm=true
  fi
done

if $missing_hbm && [[ -z "$MODEL_BASE_URL" ]]; then
  echo "ERROR: no default HBM archive is configured for $SOC."
  echo "Place matching $SOC HBM files under $GEMMA4_HOME/model, or set:"
  echo "  GEMMA4_MODEL_BASE_URL=<$SOC model directory URL>"
  exit 1
fi

if [[ -n "$MODEL_BASE_URL" ]]; then
  echo "HBM URL     : $MODEL_BASE_URL"
  for model_file in "${hbm_files[@]}"; do
    download_file "$MODEL_BASE_URL" "$model_file" "$GEMMA4_HOME/model"
  done
fi

download_file "$COMMON_MODEL_BASE_URL" tok_embeddings.bin "$GEMMA4_HOME/model"
download_file "$TOKENIZER_BASE_URL" tokenizer.json "$GEMMA4_HOME/tokenizer"
download_file "$TOKENIZER_BASE_URL" tokenizer_config.json "$GEMMA4_HOME/tokenizer"

echo "Download complete."
echo "Optional integrity check:"
echo "  sha256sum $GEMMA4_HOME/model/*.hbm"
