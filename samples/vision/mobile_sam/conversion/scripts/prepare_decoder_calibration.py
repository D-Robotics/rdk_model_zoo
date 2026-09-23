# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Prepare MobileSAM decoder calibration from a real encoder embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TARGETS = ("x5", "s100", "s100p", "s600")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MobileSAM decoder calibration tensors.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./decoder_calibration"))
    parser.add_argument("--num", type=int, default=30)
    parser.add_argument("--box", nargs=4, type=float, default=[185.0, 120.0, 380.0, 445.0])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    embedding = np.fromfile(args.embedding, dtype=np.float32).reshape(1, 256, 32, 32)
    root = args.out
    embedding_dir = root / ("calibration_embeddings" if args.target == "x5" else "image_embeddings")
    box_dir = root / ("calibration_boxes" if args.target == "x5" else "boxes")
    suffix = ".bin" if args.target == "x5" else ".npy"
    for directory in (embedding_dir, box_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob(f"*{suffix}"):
            old.unlink()
    base_box = np.array([args.box], dtype=np.float32)
    for index in range(args.num):
        scale = 1.0 + (index - args.num // 2) * 0.001
        emb = (embedding * scale).astype(np.float32)
        box = base_box.copy()
        jitter = float((index % 5) - 2)
        box[:, [0, 2]] += jitter
        box[:, [1, 3]] += jitter
        if args.target == "x5":
            emb.tofile(embedding_dir / f"emb_{index:03d}.bin")
            box.tofile(box_dir / f"box_{index:03d}.bin")
        else:
            np.save(embedding_dir / f"emb_{index:03d}.npy", emb)
            np.save(box_dir / f"box_{index:03d}.npy", box)
    print(f"Wrote {args.num} embeddings to {embedding_dir}")
    print(f"Wrote {args.num} boxes to {box_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
