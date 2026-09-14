#!/bin/bash
# Download the prebuilt HGNetV2 .bin models from the D-Robotics archive.
#
# Default: download only the b0 variant used by ``run.sh``.
# Pass variant names to download other ones, or ``all`` for the full set:
#
#     bash download.sh              # only b0 (~5.9 MB)
#     bash download.sh b3 b4        # only b3 and b4
#     bash download.sh all          # all five variants (~57 MB)
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/hgnetv2"

if [ "$#" -eq 0 ]; then
    VARIANTS=(b0)
elif [ "$1" = "all" ]; then
    VARIANTS=(b0 b1 b2 b3 b4)
else
    VARIANTS=("$@")
fi

for VARIANT in "${VARIANTS[@]}"; do
    NAME="hgnetv2_${VARIANT}_224x224_nv12.bin"
    if [ -f "${DIR}/${NAME}" ]; then
        echo "[skip] ${NAME} already present"
        continue
    fi
    echo "[download] ${NAME}"
    wget -c -P "${DIR}" "${BASE_URL}/${NAME}"
done
