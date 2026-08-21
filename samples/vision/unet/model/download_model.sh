#!/usr/bin/env bash
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESNET18_SHA256="d082ff055532081d14326d96fb2bb8ac85a0f1edc46e868cbbbea0259bc36b5f"
RESNET34_SHA256="9d758822b2de4d5aaa24b4c02479c9f742c4b4e4af075389d921c12396194ac0"
RESNET50_SHA256="22ea4eec82328d34dc963e091f3cb4e134c8a432d7c6f92d74522b998b7bd23a"
RESNET101_SHA256="04031417d3d4098bceac0bb1b731a7aed099dca5c8ee0cd30527c2f6494c7215"
RESNET152_SHA256="990855473e5411c2996bd7f161591dc7ba479402bcfe40c36d3fd2b10edbb32a"

usage() {
    cat <<'EOF'
Usage: ./download_model.sh [resnet18|resnet34|resnet50|resnet101|resnet152|all]...

With no argument, the script downloads the published ResNet18 model.
All five backbone models are published. Use 'all' to download the full family.
EOF
}

model_name() {
    case "$1" in
        resnet18|resnet34|resnet50|resnet101|resnet152)
            printf 'unet_%s_voc_512x512_nv12.bin\n' "$1"
            ;;
        *)
            return 1
            ;;
    esac
}

expected_sha256() {
    case "$1" in
        resnet18)
            printf '%s\n' "${RESNET18_SHA256}"
            ;;
        resnet34)
            printf '%s\n' "${RESNET34_SHA256}"
            ;;
        resnet50)
            printf '%s\n' "${RESNET50_SHA256}"
            ;;
        resnet101)
            printf '%s\n' "${RESNET101_SHA256}"
            ;;
        resnet152)
            printf '%s\n' "${RESNET152_SHA256}"
            ;;
        *)
            return 1
            ;;
    esac
}

verify_model() {
    local path="$1"
    local expected="$2"
    local actual

    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "SHA256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
        return 1
    fi
}

download_model() {
    local backbone="$1"
    local name
    local expected
    local destination
    local temporary

    if ! name="$(model_name "${backbone}")"; then
        echo "Unsupported backbone: ${backbone}" >&2
        usage >&2
        return 2
    fi
    expected="$(expected_sha256 "${backbone}")"
    destination="${SCRIPT_DIR}/${name}"
    temporary="${destination}.part"

    if [[ -f "${destination}" ]]; then
        verify_model "${destination}" "${expected}"
        echo "Model already exists and passed SHA256 verification: ${destination}"
        return 0
    fi

    echo "Downloading ${name}..."
    wget -c "${BASE_URL}/${name}" -O "${temporary}"
    verify_model "${temporary}" "${expected}"
    mv "${temporary}" "${destination}"
    echo "Saved ${destination}"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$#" -eq 0 ]]; then
    targets=(resnet18)
else
    targets=("$@")
fi
if [[ "${#targets[@]}" -eq 1 && "${targets[0]}" == "all" ]]; then
    targets=(resnet18 resnet34 resnet50 resnet101 resnet152)
fi

for backbone in "${targets[@]}"; do
    if [[ "${backbone}" == "all" ]]; then
        echo "The 'all' selector cannot be combined with other backbones." >&2
        exit 2
    fi
    download_model "${backbone}"
done
