#!/bin/bash
# 一次性配置 swap（需要 sudo）。建议在编译大模型前执行。
set -euo pipefail

SWAPFILE="${SWAPFILE:-/swapfile}"
SWAP_GB="${SWAP_GB:-64}"

if swapon --show | grep -q .; then
  echo "Swap already enabled:"
  swapon --show
  free -h | grep -i swap
  exit 0
fi

if [[ ! -w "$(dirname "$SWAPFILE")" ]]; then
  echo "ERROR: cannot write swap directory: $(dirname "$SWAPFILE")"
  exit 1
fi

echo "Creating ${SWAP_GB}G swap at ${SWAPFILE} (requires sudo)..."
sudo fallocate -l "${SWAP_GB}G" "$SWAPFILE"
sudo chmod 600 "$SWAPFILE"
sudo mkswap "$SWAPFILE"
sudo swapon "$SWAPFILE"

# 编译场景：内存吃紧时更愿意用 swap，避免直接 OOM kill
if [[ -w /proc/sys/vm/swappiness ]]; then
  echo 60 | sudo tee /proc/sys/vm/swappiness >/dev/null
fi

echo "Swap enabled:"
swapon --show
free -h
