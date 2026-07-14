#!/bin/bash
set -euo pipefail

GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"
MODEL_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/gemma4_e2b/model"
TOKENIZER_BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/gemma4_e2b/tokenizer"

echo "GEMMA4_HOME : $GEMMA4_HOME"
echo "Model URL   : $MODEL_BASE_URL"

if [[ -f "$GEMMA4_HOME/model/gemma4-e2b_vit_ptq.hbm" && \
      -f "$GEMMA4_HOME/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm" && \
      -f "$GEMMA4_HOME/model/tok_embeddings.bin" && \
      -f "$GEMMA4_HOME/tokenizer/tokenizer.json" && \
      -f "$GEMMA4_HOME/tokenizer/tokenizer_config.json" ]]; then
  echo "Model and tokenizer files already present, skip download."
else
  mkdir -p "$GEMMA4_HOME/model" "$GEMMA4_HOME/tokenizer"
  for model_file in \
    gemma4-e2b_vit_ptq.hbm \
    tok_embeddings.bin \
    gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm; do
    echo "Downloading $model_file..."
    wget -c -P "$GEMMA4_HOME/model" "$MODEL_BASE_URL/$model_file"
  done
  for tokenizer_file in tokenizer.json tokenizer_config.json; do
    echo "Downloading $tokenizer_file..."
    wget -c -P "$GEMMA4_HOME/tokenizer" "$TOKENIZER_BASE_URL/$tokenizer_file"
  done
fi

echo "Download complete."
echo "Optional integrity check:"
echo "  sha256sum $GEMMA4_HOME/model/*.hbm"
