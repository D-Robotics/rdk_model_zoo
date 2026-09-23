# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Export the MobileSAM image encoder ONNX shared by X5 and RDK-S."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MobileSAM image encoder ONNX.")
    parser.add_argument("--target", choices=("x5", "s100", "s100p", "s600"), default="x5")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/MobileSAM"))
    parser.add_argument("--weights", type=Path, default=Path("./workspace/MobileSAM/weights/mobile_sam.pt"))
    parser.add_argument("--output", type=Path, default=Path("./mobile_sam_image_encoder_norm_512_op11.onnx"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=11)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    sys.path.insert(0, str(args.repo))
    from ultralytics.models.sam.build import build_mobile_sam

    model = build_mobile_sam(str(args.weights)).eval()
    model.set_imgsz((args.size, args.size))
    encoder = model.image_encoder
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(encoder, dummy, str(args.output), export_params=True, opset_version=args.opset,
                          do_constant_folding=True, input_names=["normalized_images"],
                          output_names=["image_embeddings"], dynamic_axes=None)
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
