# Copyright (c) 2026 D-Robotics Corporation
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

"""Export the FAIR DINOv2 ViT-S/14 backbone to ONNX for Nash PTQ.

The exported graph uses only BPU-friendly operators (Conv / MatMul / Softmax /
LayerNormalization / Erf / elementwise). Two source-model constructs are
rewritten for the toolchain:

1. The positional-embedding bicubic `Resize` is executed once in PyTorch and
   baked into `model.pos_embed` as a constant, so no Resize node is traced.
2. `MemEffAttention` (SDPA / xformers) is monkey-patched with an explicit
   MatMul + Softmax + MatMul implementation, which lowers to fully BPU ops
   after quantization.

Run inside the OE 3.7.0 docker image (torch 2.6, legacy ONNX exporter) or on
any torch >= 2.4 host. `dynamo=False` keeps the emitted IR at version 8, the
version accepted by hb_compile 3.5.3 / hmct 2.6.5.

Example:
    python3 export_dinov2.py \
        --weights ./dinov2_vits14_pretrain.pth \
        --repo /path/to/dinov2 \
        --out ./dinov2_vits14_224.onnx
"""

from __future__ import annotations

import argparse
import os
import types

# Force the non-xformers code path before dinov2 is imported (the environment
# variable is read at import time by dinov2/layers/attention.py).
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"

IMG_SIZE = 224
PATCH_SIZE = 14
GRID = IMG_SIZE // PATCH_SIZE  # 16
NUM_TOKENS = GRID * GRID + 1  # 257 (1 cls + 256 patch)


def build_model(repo: str, weights: str) -> nn.Module:
    """Build the DINOv2 ViT-S/14 model and load the pretrain checkpoint."""

    import sys
    sys.path.insert(0, os.path.abspath(repo))
    from dinov2.models.vision_transformer import vit_small

    model = vit_small(
        patch_size=PATCH_SIZE,
        img_size=518,  # native training resolution: pos_embed matches ckpt
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )
    sd = torch.load(weights, map_location="cpu")
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    # dinov2_*_pretrain.pth ships clean unprefixed keys (cls_token, blocks.*).
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def bake_pos_embed(model: nn.Module) -> None:
    """Interpolate pos_embed to 224 once and bake it as a constant."""

    with torch.no_grad():
        dummy = torch.zeros(1, NUM_TOKENS, model.embed_dim)
        pos = model.interpolate_pos_encoding(dummy, IMG_SIZE, IMG_SIZE)
    assert pos.shape == (1, NUM_TOKENS, model.embed_dim), pos.shape
    model.register_buffer("pos_embed_baked", pos.clone())

    def prepare_tokens_static(self, x, masks=None):
        assert masks is None, "mask tokens are training-only"
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed_baked
        return x

    model.prepare_tokens_with_masks = types.MethodType(prepare_tokens_static, model)


def patch_attention(model: nn.Module) -> None:
    """Rewrite MemEffAttention.forward as explicit MatMul + Softmax."""

    def forward_export(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)  # explicit axis 2: Split with axis=-1 is not supported
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, N, Dh)
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

    for blk in model.blocks:
        blk.attn.forward = types.MethodType(forward_export, blk.attn)


class ExportWrapper(nn.Module):
    """Return cls embedding and patch tokens as two graph outputs."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, image):
        f = self.m.forward_features(image)
        return f["x_norm_clstoken"], f["x_norm_patchtokens"]


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 ViT-S/14 ONNX export")
    parser.add_argument("--weights", type=str, required=True, help="Path to dinov2_vits14_pretrain.pth (or the URL).")
    parser.add_argument("--repo", type=str, default=".", help="Path to a local clone of facebookresearch/dinov2.")
    parser.add_argument("--out", type=str, default="dinov2_vits14_224.onnx", help="Output ONNX path.")
    args = parser.parse_args()

    model = build_model(args.repo, args.weights)
    bake_pos_embed(model)
    patch_attention(model)
    wrapper = ExportWrapper(model).eval()

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        cls_t, patch_t = wrapper(x)

    torch.onnx.export(
        wrapper,
        (x,),
        args.out,
        input_names=["input"],
        output_names=["cls_feat", "patch_feat"],
        opset_version=17,
        dynamo=False,  # legacy exporter emits IR v8 (hbdk 4.7.5 limit)
        do_constant_folding=True,
    )
    print(f"[export] wrote {args.out}")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    cls_o, patch_o = sess.run(None, {"input": x.numpy()})
    for name, a, b in (("cls", cls_t.numpy(), cls_o), ("patch", patch_t.numpy(), patch_o)):
        cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"[parity] {name}: cosine={cos:.6f} max_abs={np.abs(a - b).max():.2e}")

    import onnx
    m = onnx.load(args.out)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[onnx] ir={m.ir_version} opset={m.opset_import[0].version} ops={dict(sorted(ops.items()))}")


if __name__ == "__main__":
    main()
