#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-./datasets}
mkdir -p "$DATA_ROOT/wikitext2-calibration-train" "$DATA_ROOT/wikitext2-test"
BASE=https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1
download() {
  local source=$1 destination=$2 checksum=$3
  if [[ -f "$destination" ]] && printf '%s  %s\n' "$checksum" "$destination" | sha256sum -c --status; then
    return
  fi
  curl --fail --location --retry 3 -o "$destination.part" "$BASE/$source"
  printf '%s  %s\n' "$checksum" "$destination.part" | sha256sum -c -
  mv "$destination.part" "$destination"
}
# The SDK discovers test-*.parquet; the renamed file still contains TRAIN data.
download train-00000-of-00001.parquet "$DATA_ROOT/wikitext2-calibration-train/test-train-source.parquet" e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7
download test-00000-of-00001.parquet "$DATA_ROOT/wikitext2-test/test-00000-of-00001.parquet" 5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91
