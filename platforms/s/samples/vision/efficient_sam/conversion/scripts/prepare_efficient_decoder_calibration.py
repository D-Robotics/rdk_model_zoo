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

"""Prepare EfficientSAM fixed-prompt decoder calibration featuremaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    """Create decoder calibration embeddings from one raw encoder embedding."""

    parser = argparse.ArgumentParser(description="Prepare EfficientSAM decoder calibration tensors.")
    parser.add_argument("--embedding", type=Path, required=True, help="Raw float32 encoder embedding, shape 1x256x32x32.")
    parser.add_argument("--out", type=Path, default=Path("./decoder_calibration"), help="Output calibration root.")
    parser.add_argument("--num", type=int, default=30, help="Number of calibration samples.")
    args = parser.parse_args()

    embedding = np.fromfile(args.embedding, dtype=np.float32).reshape(1, 256, 32, 32)
    embedding_dir = args.out / "image_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for old_file in embedding_dir.glob("*.npy"):
        old_file.unlink()
    for index in range(args.num):
        scale = 1.0 + (index - args.num // 2) * 0.001
        np.save(embedding_dir / f"emb_{index:03d}.npy", (embedding * scale).astype(np.float32))
    print(f"Wrote {args.num} embeddings to {embedding_dir}")


if __name__ == "__main__":
    main()
