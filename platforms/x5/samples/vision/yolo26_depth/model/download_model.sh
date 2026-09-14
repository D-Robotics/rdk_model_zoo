#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
MODEL_DIR="$ROOT/bayes-e"

# Official archive URL; override with MODEL_BASE_URL only when using an internal mirror.
MODEL_BASE_URL=${MODEL_BASE_URL:-"https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yolo26_depth"}

declare -A HASHES=(
  [n]="e55091eb594e20e37e6c36a36cce42a94ad80ec651ae893a2143cd2273ed9b0b"
  [s]="0e43958195f504d7a8ac48b1c99f4802cd9a4c3580321bfb251d0e0f892ccf4c"
  [m]="f4f2f1958dc16324932b4492490209c817cf7565c3c29240bcf4f0012f9c0be0"
  [l]="6a5fa40bda20ee56208ca6e594ecfd9781329385d0baf1b15c9eaa9625286d14"
  [x]="61798227fb7e0772a739b483ae5b5acd58a8e785dd7fd9aec5dcac7db0903d91"
)

variants=("$@")
if [[ ${#variants[@]} -eq 0 ]]; then
  variants=(n s m l x)
fi

mkdir -p "$MODEL_DIR"
for variant in "${variants[@]}"; do
  if [[ -z "${HASHES[$variant]:-}" ]]; then
    echo "Unknown variant: $variant (expected n, s, m, l, or x)" >&2
    exit 2
  fi

  name="yolo26${variant}_depth_bayese_768x768_nv12.bin"
  model="$MODEL_DIR/$name"
  if [[ ! -f "$model" ]]; then
    curl -L --fail --retry 3 "$MODEL_BASE_URL/$name" -o "$model"
  fi

  actual=$(sha256sum "$model" | awk '{print $1}')
  if [[ "$actual" != "${HASHES[$variant]}" ]]; then
    echo "SHA256 mismatch for $name" >&2
    echo "expected ${HASHES[$variant]}" >&2
    echo "actual   $actual" >&2
    exit 1
  fi
  echo "$model"
done
