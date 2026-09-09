#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${MODEL_DIR:-"$HERE/s600"}
URL=${MINICPM5_MODEL_URL:-https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz}
ARCHIVE_SHA256=8f2bef6fc7d2290f05055570e7dcfb1cf4e8c07c9a6bb0d0acc24dd202a83841
MANIFEST_SHA256=a938fb43fd7e2c161e188a12e2925f386809adcdd3ab0f0531faaa43d5affcb5

verify_model() {
  local directory=$1
  [[ -f "$directory/SHA256SUMS" ]] || return 1
  printf '%s  %s\n' "$MANIFEST_SHA256" "$directory/SHA256SUMS" | sha256sum -c --status || return 1
  # Preserve the pinned manifest bytes; accept Windows CRLF when parsing it.
  (cd "$directory" && tr -d '\r' < SHA256SUMS | sha256sum -c --quiet -)
}

if verify_model "$MODEL_DIR"; then
  echo "Verified MiniCPM5 S600 model: $MODEL_DIR"
  exit 0
fi
if [[ -e "$MODEL_DIR" ]]; then
  echo "ERROR: $MODEL_DIR exists but does not match this release; choose a new MODEL_DIR." >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$MODEL_DIR")"
STAGING=$(mktemp -d "$(dirname -- "$MODEL_DIR")/.minicpm5-download.XXXXXX")
trap 'rm -rf -- "$STAGING"' EXIT
curl --fail --location --retry 3 --output "$STAGING/model.tar.gz.part" "$URL"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$STAGING/model.tar.gz.part" | sha256sum -c -
tar -xzf "$STAGING/model.tar.gz.part" -C "$STAGING"
verify_model "$STAGING/minicpm5-2b-s600"
mv -- "$STAGING/minicpm5-2b-s600" "$MODEL_DIR"
echo "Downloaded and verified MiniCPM5 S600 model: $MODEL_DIR"
