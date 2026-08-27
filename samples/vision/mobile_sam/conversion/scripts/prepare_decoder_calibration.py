# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare MobileSAM decoder calibration featuremaps.

The decoder is calibrated from real encoder embeddings. Generate one embedding
on board or with the floating encoder, then use this script to create >=20 raw
featuremap samples plus matching box prompt samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    """Create decoder calibration embeddings and box prompt tensors."""

    parser = argparse.ArgumentParser(description="Prepare MobileSAM decoder calibration tensors.")
    parser.add_argument("--embedding", type=Path, required=True, help="Raw float32 encoder embedding, shape 1x256x32x32.")
    parser.add_argument("--out", type=Path, default=Path("./decoder_calibration"), help="Output calibration root.")
    parser.add_argument("--num", type=int, default=30, help="Number of calibration samples.")
    parser.add_argument("--box", nargs=4, type=float, default=[185.0, 120.0, 380.0, 445.0], help="Base box x1 y1 x2 y2.")
    args = parser.parse_args()

    embedding = np.fromfile(args.embedding, dtype=np.float32).reshape(1, 256, 32, 32)
    box = np.array([args.box], dtype=np.float32)
    embedding_dir = args.out / "image_embeddings"
    box_dir = args.out / "boxes"
    for directory in (embedding_dir, box_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old_file in directory.glob("*.npy"):
            old_file.unlink()
    for index in range(args.num):
        scale = 1.0 + (index - args.num // 2) * 0.001
        embedding_i = (embedding * scale).astype(np.float32)
        box_i = box.copy()
        jitter = float((index % 5) - 2)
        box_i[:, [0, 2]] += jitter
        box_i[:, [1, 3]] += jitter
        np.save(embedding_dir / f"emb_{index:03d}.npy", embedding_i)
        np.save(box_dir / f"box_{index:03d}.npy", box_i)
    print(f"Wrote {args.num} embeddings to {embedding_dir}")
    print(f"Wrote {args.num} boxes to {box_dir}")


if __name__ == "__main__":
    main()
