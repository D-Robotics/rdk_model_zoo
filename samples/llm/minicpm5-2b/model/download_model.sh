#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BOARD=${BOARD:-s600}
REMOTE_DIR=rdk_s100
case "$BOARD" in
  s600)
    REMOTE_DIR=rdk_s600
    ARCHIVE=minicpm5-2b_s600_oellm2_w8_ctx4096_20260908.tar.gz
    ARCHIVE_SHA256=8f2bef6fc7d2290f05055570e7dcfb1cf4e8c07c9a6bb0d0acc24dd202a83841
    MANIFEST_SHA256=a938fb43fd7e2c161e188a12e2925f386809adcdd3ab0f0531faaa43d5affcb5
    ;;
  s100)
    ARCHIVE=minicpm5-2b_s100_oellm1_w8_ctx4096_20260909.tar.gz
    ARCHIVE_SHA256=dc130ae21dc1f1cc266c526e090b03b4d3be4af8cf4157c4b5841b7f47ab818d
    MANIFEST_SHA256=aaa7e2786e3c01d68d6d8e1f3ace8892f9faaddbede9af2a47c9202d8e819bd9
    ;;
  s100p)
    ARCHIVE=minicpm5-2b_s100p_oellm1_w8_ctx4096_20260909.tar.gz
    ARCHIVE_SHA256=65c89ae7ead48711a392fa556f602358ffffe16438e94051d1f8cca46d4e096a
    MANIFEST_SHA256=ead63bc56816501162f8b6d52b2bcf611710663bca10ab84711d0eed7e3191ce
    ;;
  *) echo "BOARD must be s100, s100p or s600" >&2; exit 2;;
esac
MODEL_DIR=${MODEL_DIR:-"$HERE/$BOARD"}
URL=${MINICPM5_MODEL_URL:-https://archive.d-robotics.cc/downloads/rdk_model_zoo/$REMOTE_DIR/$ARCHIVE}

verify_model() {
  local directory=$1
  [[ -f "$directory/SHA256SUMS" ]] || return 1
  printf '%s  %s\n' "$MANIFEST_SHA256" "$directory/SHA256SUMS" | sha256sum -c --status || return 1
  # Preserve the pinned manifest bytes; accept Windows CRLF when parsing it.
  (cd "$directory" && tr -d '\r' < SHA256SUMS | sha256sum -c --quiet -)
}

if verify_model "$MODEL_DIR"; then
  echo "Verified MiniCPM5 $BOARD model: $MODEL_DIR"
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
verify_model "$STAGING/minicpm5-2b-$BOARD"
mv -- "$STAGING/minicpm5-2b-$BOARD" "$MODEL_DIR"
echo "Downloaded and verified MiniCPM5 $BOARD model: $MODEL_DIR"
