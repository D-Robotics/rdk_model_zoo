#!/bin/bash
# Explicit dependency preparation; neither launcher nor CMake invokes this.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/tokenizers-cpp"
COMMIT="c586c52f93f7b060753bd2388eb96a105cb7374d"
URL="https://github.com/mlc-ai/tokenizers-cpp.git"
case "${1:-}" in
  --help|-h) echo "Usage: bash install_tokenizers_cpp.sh [--dry-run]"; exit 0 ;;
  --dry-run)
    [[ $# == 1 ]] || { echo "Unexpected arguments" >&2; exit 2; }
    echo "Prepare $URL @ $COMMIT -> $DEST"
    echo "Requires Git and Rust >= 1.80; no toolchain installation. Existing trees are checked, never replaced."
    exit 0 ;;
  '') ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
export PATH="$HOME/.cargo/bin:$PATH"
for tool in git rustc cargo; do
  command -v "$tool" >/dev/null || { echo "Missing $tool; install dependencies explicitly (see third_party/README.md)." >&2; exit 2; }
done
version="$(rustc --version | awk '{print $2}')"
if [[ ! "$version" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] ||
   (( BASH_REMATCH[1] < 1 || (BASH_REMATCH[1] == 1 && BASH_REMATCH[2] < 80) )); then
  echo "Rust >= 1.80 stable required; found $version. Install a compatible toolchain explicitly." >&2
  exit 2
fi

verify_checkout() {
  local tree="$1" status
  [[ -d "$tree/.git" && -f "$tree/CMakeLists.txt" && -f "$tree/msgpack/CMakeLists.txt" && -f "$tree/sentencepiece/CMakeLists.txt" ]] || {
    echo "Incomplete existing dependency: $tree; preserve it and choose a clean preparation location." >&2; return 1;
  }
  [[ "$(git -C "$tree" rev-parse HEAD)" == "$COMMIT" ]] || { echo "Dependency commit mismatch; existing tree preserved." >&2; return 1; }
  status="$(git -C "$tree" status --porcelain --untracked-files=all)"
  # The only permitted local change is the documented lockfile-format patch.
  if [[ -n "$status" ]]; then
    [[ "$status" == ' M rust/Cargo.lock' ]] &&
      cmp -s "$tree/rust/Cargo.lock" <(git -C "$tree" show HEAD:rust/Cargo.lock | awk 'NR<=5 && /^version = 4$/ {sub(/4/,"3")} {print}') || {
        echo "Dependency has local changes; preserved without replacement." >&2; return 1;
      }
  fi
  status="$(git -C "$tree" submodule status --recursive)"
  if [[ -n "$status" ]] && printf '%s\n' "$status" | grep -q '^[+-U]'; then
    echo "Dependency submodules are uninitialized or differ from their pins." >&2; return 1
  fi
  git -C "$tree" submodule foreach --quiet --recursive 'test -z "$(git status --porcelain --untracked-files=all)"'
}
normalize_lock() {
  local tree="$1" temporary
  if [[ -f "$tree/rust/Cargo.lock" ]]; then
    temporary="$(mktemp "$tree/rust/Cargo.lock.XXXXXX")"
    awk 'NR<=5 && /^version = 4$/ {sub(/4/,"3")} {print}' "$tree/rust/Cargo.lock" > "$temporary"
    mv "$temporary" "$tree/rust/Cargo.lock"
  fi
}
if [[ -e "$DEST" || -L "$DEST" ]]; then
  [[ ! -L "$DEST" ]] || { echo "Dependency destination is a symlink; preserved." >&2; exit 2; }
  verify_checkout "$DEST"
  normalize_lock "$DEST"
  echo "tokenizers-cpp already prepared at $DEST @ $COMMIT"
  exit 0
fi
# Work only in a newly created staging directory. Failed preparation never
# destroys a pre-existing destination; trap removes only this invocation's stage.
STAGE="$(mktemp -d "$SCRIPT_DIR/.tokenizers-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
git clone --depth 1 "$URL" "$STAGE/source"
git -C "$STAGE/source" fetch --depth 1 origin "$COMMIT"
git -C "$STAGE/source" checkout --detach "$COMMIT"
git -C "$STAGE/source" submodule update --init --recursive --depth 1
verify_checkout "$STAGE/source"
normalize_lock "$STAGE/source"
[[ ! -e "$DEST" && ! -L "$DEST" ]] || { echo "Destination appeared during preparation; preserved." >&2; exit 2; }
mv "$STAGE/source" "$DEST"
echo "tokenizers-cpp ready at $DEST @ $COMMIT"
