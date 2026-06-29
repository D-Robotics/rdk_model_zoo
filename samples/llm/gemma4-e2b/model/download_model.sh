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
  echo "ERROR: huggingface_hub CLI (hf) not found."
  echo
  echo "Install it once, then re-run ./run.sh:"
  echo "  python3 -m pip install --user 'huggingface_hub>=0.26.0'"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  echo
  echo "Or on systems with PEP 668 restrictions:"
  echo "  python3 -m pip install --user --break-system-packages 'huggingface_hub>=0.26.0'"
  exit 1
fi

echo "Downloading pre-compiled HBM models from HuggingFace..."
hf download "$REPO_ID" --local-dir "$GEMMA4_HOME"

echo "Download complete."
echo "Optional integrity check:"
echo "  sha256sum $GEMMA4_HOME/model/*.hbm"
