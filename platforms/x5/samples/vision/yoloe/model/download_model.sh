#!/bin/bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: bash download_model.sh [s|m|l|all] (default: s)" >&2
    exit 2
fi
SIZE="${1:-s}"
case "$SIZE" in
    s|m|l|all) ;;
    -h|--help)
        echo "Usage: bash download_model.sh [s|m|l|all] (default: s)"
        exit 0
        ;;
    *) echo "Expected size s, m, l or all" >&2; exit 2 ;;
esac
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/yoloe"
TEMP=""
trap 'if [ -n "$TEMP" ]; then rm -f -- "$TEMP"; fi' EXIT

download_model() {
    local size="$1" hash filename target
    case "$size" in
        s) hash="8b997e9148a1797a3196f6230c1db2029b85ebdd69bd39ad609577723af3f8fa" ;;
        m) hash="ec5607ba89bb981b8155db18a9cd2af9c006ade2474de5de7a0bb34185aa9e31" ;;
        l) hash="a5ab9d7912bd6d44187c1075a621bbdef36ff3717d16d3246b09f347828a9ed2" ;;
    esac
    filename="yoloe_11${size}_seg_pf_bayese_640x640_nv12.bin"
    target="${MODEL_DIR}/${filename}"
    if [ -e "$target" ] || [ -L "$target" ]; then
        if [ -f "$target" ] && printf '%s  %s\n' "$hash" "$target" | sha256sum --check --status; then
            echo "Already verified: $target"
            return
        fi
        echo "Refusing to overwrite existing file with invalid SHA256: $target" >&2
        return 1
    fi
    TEMP="$(mktemp "${target}.part.XXXXXX")"
    curl --fail --location --proto '=https' --proto-redir '=https' \
        --connect-timeout 15 --max-time 600 --retry 2 \
        --output "$TEMP" "${BASE_URL}/${filename}"
    if ! printf '%s  %s\n' "$hash" "$TEMP" | sha256sum --check --status; then
        echo "SHA256 mismatch: $filename" >&2
        return 1
    fi
    ln -- "$TEMP" "$target"
    rm -f -- "$TEMP"
    TEMP=""
    echo "Verified model: $target"
}

if [ "$SIZE" = all ]; then
    for size in s m l; do
        download_model "$size"
    done
else
    download_model "$SIZE"
fi
