#!/bin/bash
# Download and set up tokenizers-cpp (HuggingFace tokenizers C++ binding +
# sentencepiece) at a pinned commit, including its git submodules.
# Called by runtime/cpp/run.sh before the first build; safe to re-run
# (skips if already present).
#
# Network: set HTTP_PROXY/HTTPS_PROXY if git clone is slow or blocked.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/tokenizers-cpp"

# Pinned commit from mlc-ai/tokenizers-cpp (master).
# The Rust binding uses tokenizers 0.21.2 + onig, matching the reference
# OpenExplorer_LLM-s600 tokenizer stack.
COMMIT="c586c52f93f7b060753bd2388eb96a105cb7374d"

if [[ -d "$DEST" && -f "$DEST/CMakeLists.txt" && -f "$DEST/msgpack/CMakeLists.txt" ]]; then
  echo "tokenizers-cpp already present at $DEST, skip download."
  exit 0
fi

echo "Downloading tokenizers-cpp @ ${COMMIT:0:8} ..."
if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found. Install with: sudo apt install -y curl" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found. Install with: sudo apt install -y git" >&2
  exit 1
fi

# Clone at the pinned commit and fetch submodules (sentencepiece, msgpack).
# A shallow clone + submodule init keeps the download small.
rm -rf "$DEST"
git clone --depth 1 "https://github.com/mlc-ai/tokenizers-cpp.git" "$DEST"
cd "$DEST"
git fetch --depth 1 origin "$COMMIT"
git checkout "$COMMIT"
git submodule update --init --depth 1

echo "tokenizers-cpp ready at $DEST"
