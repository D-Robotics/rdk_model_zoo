#!/bin/bash
set -euo pipefail

GEMMA4_HOME="${GEMMA4_HOME:-$HOME/gemma4_e2b}"
REPO_ID="ShockleyWong/gemma4-e2b-rdk-s100p"

echo "GEMMA4_HOME : $GEMMA4_HOME"
echo "Repo        : $REPO_ID"

if [[ -f "$GEMMA4_HOME/model/gemma4-e2b_vit_ptq.hbm" && \
      -f "$GEMMA4_HOME/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm" && \
      -f "$GEMMA4_HOME/model/tok_embeddings.bin" && \
      -f "$GEMMA4_HOME/tokenizer/tokenizer.json" ]]; then
  echo "Model files already present, skip download."
  exit 0
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Installing huggingface_hub..."
  pip install -q huggingface_hub
fi

echo "Downloading pre-compiled HBM models from HuggingFace..."
hf download "$REPO_ID" --local-dir "$GEMMA4_HOME"

echo "Download complete."
echo "Optional integrity check:"
echo "  sha256sum $GEMMA4_HOME/model/*.hbm"
