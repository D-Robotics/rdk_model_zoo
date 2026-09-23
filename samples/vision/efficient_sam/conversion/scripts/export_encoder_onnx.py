# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Export the EfficientSAM image encoder for X5 or RDK-S.

This is source-derived conversion code. It is not run by the unified sample's
host tests; execute it only in the matching upstream/OE environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TARGETS = ("x5", "s100", "s100p", "s600")
DEFAULT_OUTPUTS = {
    "x5": "efficient_sam_vitt_encoder_512_splitqkv_op11.onnx",
    "s100": "efficient_sam_vitt_encoder_512_op11.onnx",
    "s100p": "efficient_sam_vitt_encoder_512_op11.onnx",
    "s600": "efficient_sam_vitt_encoder_512_op11.onnx",
}


def _build_encoder_wrapper(model, torch):
    class EfficientSAMImageEncoder(torch.nn.Module):
        """Registered module wrapper around EfficientSAM's embedding method."""

        def __init__(self, wrapped_model):
            super().__init__()
            self.model = wrapped_model

        def forward(self, batched_images):
            return self.model.get_image_embeddings(batched_images)

    return EfficientSAMImageEncoder(model).eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export EfficientSAM image encoder ONNX.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/EfficientSAM"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/EfficientSAM/weights/efficient_sam_vitt.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=11)
    return parser


def split_qkv_linear_modules(model, torch) -> None:
    """Apply the source ONNX-friendly split-QKV rewrite."""
    for block in model.image_encoder.blocks:
        qkv = block.attn.qkv
        dim = qkv.in_features
        bias = qkv.bias is not None
        q = torch.nn.Linear(dim, dim, bias=bias)
        k = torch.nn.Linear(dim, dim, bias=bias)
        v = torch.nn.Linear(dim, dim, bias=bias)
        with torch.no_grad():
            q.weight.copy_(qkv.weight[:dim])
            k.weight.copy_(qkv.weight[dim : 2 * dim])
            v.weight.copy_(qkv.weight[2 * dim :])
            if bias:
                q.bias.copy_(qkv.bias[:dim])
                k.bias.copy_(qkv.bias[dim : 2 * dim])
                v.bias.copy_(qkv.bias[2 * dim :])
        block.attn.q = q
        block.attn.k = k
        block.attn.v = v

        def forward_with_split_qkv(x, attn=block.attn):
            batch, tokens, channels = x.shape
            q = attn.q(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            k = attn.k(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            v = attn.v(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            attn_out = (q @ k.transpose(-2, -1)) * attn.scale
            attn_out = attn_out.softmax(dim=-1)
            return attn.proj((attn_out @ v).transpose(1, 2).reshape(batch, tokens, channels))

        block.attn.forward = forward_with_split_qkv


def _load_model(args):
    if args.target == "x5":
        expected = (args.repo / "weights" / "efficient_sam_vitt.pt").resolve()
        supplied = args.checkpoint.resolve()
        if supplied != expected:
            raise ValueError(f"X5 builder uses only {expected}; custom --checkpoint is unsupported")
        if not expected.is_file():
            raise FileNotFoundError(expected)
    import torch

    sys.path.insert(0, str(args.repo))
    if args.target == "x5":
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

        old_cwd = os.getcwd()
        os.chdir(args.repo)
        try:
            model = build_efficient_sam_vitt().eval()
        finally:
            os.chdir(old_cwd)
    else:
        from efficient_sam.efficient_sam import build_efficient_sam

        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        model = build_efficient_sam(192, 3, checkpoint=str(args.checkpoint)).eval()
    return torch, model


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch, model = _load_model(args)
    model.image_encoder.img_size = args.size
    model.image_encoder.image_embedding_size = args.size // 16
    split_qkv_linear_modules(model, torch)
    output = args.output or Path(DEFAULT_OUTPUTS[args.target])
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    encoder = _build_encoder_wrapper(model, torch)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy,
            str(output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["batched_images"],
            output_names=["image_embeddings"],
            dynamic_axes=None,
        )
    print(f"Exported {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
