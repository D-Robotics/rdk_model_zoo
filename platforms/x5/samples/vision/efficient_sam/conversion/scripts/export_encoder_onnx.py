"""Export EfficientSAM-Tiny image encoder ONNX for RDK X5 quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


def split_qkv_linear_modules(model: torch.nn.Module) -> None:
    """Replace EfficientSAM encoder QKV Linear modules with explicit q/k/v modules."""

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
            """Run attention with independent Q, K, and V projections for ONNX export."""

            batch, tokens, channels = x.shape
            q = attn.q(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            k = attn.k(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            v = attn.v(x).reshape(batch, tokens, attn.num_heads, channels // attn.num_heads).permute(0, 2, 1, 3)
            attn_out = (q @ k.transpose(-2, -1)) * attn.scale
            attn_out = attn_out.softmax(dim=-1)
            attn_out = (attn_out @ v).transpose(1, 2).reshape(batch, tokens, channels)
            return attn.proj(attn_out)

        block.attn.forward = forward_with_split_qkv


def main() -> None:
    """Export the EfficientSAM-Tiny image encoder ONNX model."""

    parser = argparse.ArgumentParser(description="Export EfficientSAM-Tiny encoder ONNX.")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/EfficientSAM"), help="Path to cloned EfficientSAM repository.")
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/EfficientSAM/weights/efficient_sam_vitt.pt"), help="Path to efficient_sam_vitt.pt.")
    parser.add_argument("--output", type=Path, default=Path("./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx"), help="Output ONNX path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

    old_cwd = os.getcwd()
    os.chdir(args.repo)
    try:
        model = build_efficient_sam_vitt().eval()
    finally:
        os.chdir(old_cwd)
    model.image_encoder.img_size = args.size
    model.image_encoder.image_embedding_size = args.size // 16
    split_qkv_linear_modules(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model.get_image_embeddings,
            dummy,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["batched_images"],
            output_names=["image_embeddings"],
            dynamic_axes=None,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()