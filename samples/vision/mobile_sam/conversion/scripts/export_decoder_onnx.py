# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Export the MobileSAM box-prompt decoder ONNX shared by X5 and RDK-S."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MobileSAM box-prompt decoder ONNX.")
    parser.add_argument("--target", choices=("x5", "s100", "s100p", "s600"), default="x5")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/MobileSAM"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/MobileSAM/weights/mobile_sam.pt"))
    parser.add_argument("--output", type=Path, default=Path("./mobile_sam_decoder_512_box_op11.onnx"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--box", nargs=4, type=float, default=[185.0, 120.0, 380.0, 445.0])
    return parser


def _build_decoder_wrapper(torch, prompt_encoder, mask_decoder):
    class MobileSAMDecoder(torch.nn.Module):
        """Registered wrapper preserving prompt and mask decoder parameters."""

        def __init__(self):
            super().__init__()
            self.prompt_encoder = prompt_encoder
            self.mask_decoder = mask_decoder

        def forward(self, image_embeddings, boxes):
            sparse, dense = self.prompt_encoder(points=None, boxes=boxes, masks=None)
            return self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=True,
            )

    return MobileSAMDecoder().eval()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    sys.path.insert(0, str(args.repo))
    from ultralytics.models.sam.build import build_mobile_sam

    model = build_mobile_sam(str(args.checkpoint)).eval()
    model.set_imgsz((args.size, args.size))
    prompt_encoder, mask_decoder = model.prompt_encoder, model.mask_decoder

    args.output.parent.mkdir(parents=True, exist_ok=True)
    embedding = torch.randn(1, 256, args.size // 16, args.size // 16, dtype=torch.float32)
    boxes = torch.tensor([args.box], dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(_build_decoder_wrapper(torch, prompt_encoder, mask_decoder), (embedding, boxes), str(args.output), export_params=True,
                          opset_version=args.opset, do_constant_folding=True,
                          input_names=["image_embeddings", "boxes"],
                          output_names=["low_res_masks", "iou_predictions"], dynamic_axes=None)
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
