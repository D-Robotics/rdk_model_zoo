# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Run a float EfficientSAM encoder ONNX and save one decoder-calibration embedding."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump one EfficientSAM encoder embedding.")
    parser.add_argument("--onnx", type=Path, default=Path("./efficient_sam_vitt_encoder_512_op11.onnx"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("./encoder_embedding.bin"))
    parser.add_argument("--size", type=int, default=512)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import onnxruntime as ort

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32) / 255.0
    session = ort.InferenceSession(str(args.onnx))
    embedding = session.run(["image_embeddings"], {"batched_images": tensor})[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embedding.astype(np.float32).tofile(str(args.output))
    print(f"Wrote {args.output} (shape {embedding.shape}, dtype float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
