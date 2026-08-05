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
MIN_RUST_VERSION="1.80.0"

version_at_least() {
  local current="$1"
  local required="$2"
  [[ "$(printf '%s\n' "$required" "$current" | sort -V | head -n 1)" == "$required" ]]
}

ensure_compatible_rust() {
  export PATH="$HOME/.cargo/bin:$PATH"

  local current=""
  if command -v rustc >/dev/null 2>&1; then
    current="$(rustc --version 2>/dev/null | awk '{print $2}' || true)"
  fi
  if [[ -n "$current" ]] && version_at_least "$current" "$MIN_RUST_VERSION"; then
    echo "Rust $current already satisfies >= $MIN_RUST_VERSION."
    return
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: Rust >= $MIN_RUST_VERSION is required and curl is unavailable." >&2
    echo "Install curl, then rerun this script." >&2
    exit 1
  fi

  echo "Installing a current Rust toolchain (found: ${current:-none}) ..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
  export PATH="$HOME/.cargo/bin:$PATH"

  current="$(rustc --version | awk '{print $2}')"
  if ! version_at_least "$current" "$MIN_RUST_VERSION"; then
    echo "ERROR: Rust upgrade failed; found $current, need >= $MIN_RUST_VERSION." >&2
    exit 1
  fi
  echo "Rust $current is ready."
}

normalize_cargo_lock() {
  local lock_file="$DEST/rust/Cargo.lock"
  if [[ -f "$lock_file" ]] && grep -q '^version = 4' "$lock_file"; then
    sed -i '1,5s/^version = 4/version = 3/' "$lock_file"
    echo "Normalized Cargo.lock to stable lockfile version 3."
  fi
}

ensure_compatible_rust

if [[ -d "$DEST" && -f "$DEST/CMakeLists.txt" && -f "$DEST/msgpack/CMakeLists.txt" ]]; then
  normalize_cargo_lock
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
normalize_cargo_lock

echo "tokenizers-cpp ready at $DEST"
