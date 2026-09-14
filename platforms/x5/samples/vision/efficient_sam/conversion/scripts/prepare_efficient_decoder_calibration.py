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
    embedding_dir = args.out / "calibration_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    for old_file in embedding_dir.glob("*.bin"):
        old_file.unlink()
    for index in range(args.num):
        scale = 1.0 + (index - args.num // 2) * 0.001
        (embedding * scale).astype(np.float32).tofile(embedding_dir / f"emb_{index:03d}.bin")
    print(f"Wrote {args.num} embeddings to {embedding_dir}")


if __name__ == "__main__":
    main()