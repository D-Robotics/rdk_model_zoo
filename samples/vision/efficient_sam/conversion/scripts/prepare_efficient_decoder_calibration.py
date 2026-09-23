# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Prepare EfficientSAM decoder calibration from a real encoder embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TARGETS = ("x5", "s100", "s100p", "s600")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare EfficientSAM decoder calibration tensors.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--embedding", type=Path, required=True, help="Raw float32 1x256x32x32 encoder embedding.")
    parser.add_argument("--out", type=Path, default=Path("./decoder_calibration"))
    parser.add_argument("--num", type=int, default=30)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    embedding = np.fromfile(args.embedding, dtype=np.float32).reshape(1, 256, 32, 32)
    directory = args.out / ("calibration_embeddings" if args.target == "x5" else "image_embeddings")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".bin" if args.target == "x5" else ".npy"
    for old_file in directory.glob(f"*{suffix}"):
        old_file.unlink()
    for index in range(args.num):
        value = (embedding * (1.0 + (index - args.num // 2) * 0.001)).astype(np.float32)
        if args.target == "x5":
            value.tofile(directory / f"emb_{index:03d}.bin")
        else:
            np.save(directory / f"emb_{index:03d}.npy", value)
    print(f"Wrote {args.num} embeddings to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
