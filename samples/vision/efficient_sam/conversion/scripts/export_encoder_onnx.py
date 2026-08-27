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

"""Export EfficientSAM-Tiny image encoder ONNX for RDK-S quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


class EfficientSAMImageEncoder(torch.nn.Module):
    """Expose EfficientSAM's image-embedding method as an ONNX module."""

    def __init__(self, model: torch.nn.Module):
        """Initialize the encoder wrapper around an EfficientSAM encoder model.

        Args:
            model: EfficientSAM model exposing ``get_image_embeddings``.
        """

        super().__init__()
        self.model = model

    def forward(self, batched_images: torch.Tensor) -> torch.Tensor:
        """Return image embeddings for a normalized RGB batch."""
        return self.model.get_image_embeddings(batched_images)


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
    parser.add_argument("--output", type=Path, default=Path("./efficient_sam_vitt_encoder_512_op11.onnx"), help="Output ONNX path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from efficient_sam.efficient_sam import build_efficient_sam

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    model = build_efficient_sam(192, 3, checkpoint=str(args.checkpoint)).eval()
    model.image_encoder.img_size = args.size
    model.image_encoder.image_embedding_size = args.size // 16
    split_qkv_linear_modules(model)
    encoder = EfficientSAMImageEncoder(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
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
