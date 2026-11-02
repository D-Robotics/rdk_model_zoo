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
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import types
import urllib.parse
import urllib.request

# Force the non-xformers code path before dinov2 is imported (the environment
# variable is read at import time by dinov2/layers/attention.py).
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DINOV2_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
RESIZE_SIZE = 256

IMAGE_SIZE = 224
PATCH_SIZE = 14
GRID = IMAGE_SIZE // PATCH_SIZE  # 16
NUM_TOKENS = GRID * GRID + 1  # 257 (1 cls + 256 patch)


def _sha256_file(path: str) -> str:
    """Return the SHA-256 digest for a file without loading it into memory."""

    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_checkpoint(path: str, source: str, expected_sha256: str):
    """Check a checkpoint digest before deserializing it with PyTorch."""

    actual_sha256 = _sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Checkpoint SHA-256 mismatch for "
            f"{source}: expected {expected_sha256}, got {actual_sha256}."
        )
    return torch.load(path, map_location="cpu")


def load_checkpoint(weights: str, expected_sha256: str):
    """Load a locally stored or HTTP(S)-hosted checkpoint after validation.

    Args:
        weights: Local checkpoint path or HTTP(S) checkpoint URL.
        expected_sha256: SHA-256 digest required for the complete checkpoint.

    Returns:
        Any: The checkpoint object deserialized with ``torch.load``.

    Raises:
        ValueError: If the checkpoint digest differs from ``expected_sha256``.
        OSError: If the local file or remote checkpoint cannot be read.
    """

    scheme = urllib.parse.urlparse(weights).scheme.lower()
    if scheme not in {"http", "https"}:
        return _load_verified_checkpoint(weights, weights, expected_sha256)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(prefix="dinov2-weights-", suffix=".pth", delete=False) as temp_file:
            temp_path = temp_file.name
            with urllib.request.urlopen(weights) as response:
                while chunk := response.read(1024 * 1024):
                    temp_file.write(chunk)
        return _load_verified_checkpoint(temp_path, weights, expected_sha256)
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


def _checkout_command(repo: str, revision: str) -> str:
    """Format a shell-ready checkout command for the current platform."""

    command = ["git", "-C", repo, "checkout", revision]
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def verify_repo_revision(repo: str, expected_revision: str) -> None:
    """Require a DINOv2 source checkout to match the pinned revision.

    Args:
        repo: Path to the local DINOv2 git repository.
        expected_revision: Full commit SHA required for export.

    Raises:
        RuntimeError: If ``repo`` is not checked out at ``expected_revision``.
    """

    checkout = _checkout_command(repo, expected_revision)
    try:
        actual_revision = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Cannot determine DINOv2 source revision for {repo}. Run: {checkout}"
        ) from exc
    if actual_revision != expected_revision:
        raise RuntimeError(
            "DINOv2 source revision mismatch: "
            f"expected {expected_revision}, got {actual_revision}. Run: {checkout}"
        )


def build_model(repo: str, checkpoint) -> nn.Module:
    """Build the DINOv2 ViT-S/14 model and load a verified checkpoint.

    Args:
        repo: Path to the checked-out DINOv2 source repository.
        checkpoint: Verified checkpoint object returned by ``load_checkpoint``.

    Returns:
        nn.Module: Evaluation-mode DINOv2 ViT-S/14 backbone.
    """

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
    sd = checkpoint
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    # dinov2_*_pretrain.pth ships clean unprefixed keys (cls_token, blocks.*).
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def bake_pos_embed(model: nn.Module) -> None:
    """Interpolate pos_embed to 224 once and bake it as a constant.

    Args:
        model: DINOv2 backbone whose positional embedding is rewritten.
    """

    with torch.no_grad():
        dummy = torch.zeros(1, NUM_TOKENS, model.embed_dim)
        pos = model.interpolate_pos_encoding(dummy, IMAGE_SIZE, IMAGE_SIZE)
    assert pos.shape == (1, NUM_TOKENS, model.embed_dim), pos.shape
    model.register_buffer("pos_embed_baked", pos.clone())

    def prepare_tokens_static(self, x, masks=None):
        # Static positional embeddings keep Resize out of the traced graph.
        assert masks is None, "mask tokens are training-only"
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed_baked
        return x

    model.prepare_tokens_with_masks = types.MethodType(prepare_tokens_static, model)


def patch_attention(model: nn.Module) -> None:
    """Rewrite MemEffAttention.forward as explicit MatMul plus Softmax.

    Args:
        model: DINOv2 backbone whose attention layers are patched for export.
    """

    def forward_export(self, x, attn_bias=None):
        # Explicit attention avoids unsupported SDPA and xformers export ops.
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
        """Initialize the wrapper around an evaluation DINOv2 backbone.

        Args:
            m: Model that supplies ``forward_features`` for an input image.
        """

        super().__init__()
        self.m = m

    def forward(self, image):
        """Return normalized CLS and patch-token embeddings for an image.

        Args:
            image: NCHW float tensor with a 224 by 224 spatial shape.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: CLS embedding and patch tokens.
        """

        f = self.m.forward_features(image)
        return f["x_norm_clstoken"], f["x_norm_patchtokens"]


def check_parity(reference, actual, *, stage, sample):
    """Require finite, shape-correct and numerically close embedding outputs."""
    if len(reference) != 2 or len(actual) != 2:
        raise ValueError(f"{stage}/{sample}: expected both embedding outputs")
    records = []
    for name, shape, a, b in zip(("cls_feat", "patch_feat"), ((1, 384), (1, 256, 384)), reference, actual):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"{stage}/{sample}/{name}: invalid shape or non-finite output")
        np.testing.assert_allclose(b, a, atol=1e-3, rtol=1e-3,
                                   err_msg=f"{stage}/{sample}/{name}")
        af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
        cosine = float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
        if cosine < 0.99999:
            raise ValueError(f"{stage}/{sample}/{name}: cosine {cosine} below 0.99999")
        records.append({"stage": stage, "sample": sample, "output": name,
                        "cosine": cosine, "max_abs": float(np.abs(a - b).max())})
    return records


def main() -> None:
    """Export the pinned DINOv2 checkpoint to ONNX and report parity.

    Raises:
        RuntimeError: If the source checkout is not at the required revision.
        ValueError: If the checkpoint SHA-256 is invalid.
    """

    parser = argparse.ArgumentParser(description="DINOv2 ViT-S/14 ONNX export")
    parser.add_argument("--weights", type=str, required=True, help="Path to dinov2_vits14_pretrain.pth (or the URL).")
    parser.add_argument("--weights-sha256", type=str, default=WEIGHTS_SHA256, help="Expected SHA-256 for the checkpoint.")
    parser.add_argument("--repo", type=str, default=".", help="Path to a local clone of facebookresearch/dinov2.")
    parser.add_argument("--repo-revision", type=str, default=DINOV2_SOURCE_REVISION, help="Required git revision for the DINOv2 source clone.")
    parser.add_argument("--out", type=str, default="dinov2_vits14_224.onnx", help="Output ONNX path.")
    parser.add_argument("--parity-input", action="append", default=[], help="Normalized float32 NCHW NPY from real calibration images; repeatable.")
    parser.add_argument("--parity-report", help="Write the successful upstream/rewrite/ONNX comparison report.")
    args = parser.parse_args()

    verify_repo_revision(args.repo, args.repo_revision)
    checkpoint = load_checkpoint(args.weights, args.weights_sha256)
    model = build_model(args.repo, checkpoint)
    torch.manual_seed(0)
    inputs = [("random_seed_0", torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))]
    for path in args.parity_input:
        tensor = np.load(path, allow_pickle=False)
        if tensor.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError(f"Invalid normalized parity input: {path}")
        inputs.append((path, torch.from_numpy(tensor)))
    with torch.no_grad():
        upstream = [[value.numpy() for value in ExportWrapper(model)(tensor)] for _, tensor in inputs]
    bake_pos_embed(model)
    patch_attention(model)
    wrapper = ExportWrapper(model).eval()

    x = inputs[0][1]
    with torch.no_grad():
        rewritten = [[value.numpy() for value in wrapper(tensor)] for _, tensor in inputs]
    comparisons = []
    for (sample, _), a, b in zip(inputs, upstream, rewritten):
        comparisons.extend(check_parity(a, b, stage="upstream_to_rewrite", sample=sample))

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
    for (sample, tensor), reference in zip(inputs, upstream):
        actual = sess.run(None, {"input": tensor.numpy()})
        comparisons.extend(check_parity(reference, actual, stage="upstream_to_onnx", sample=sample))
    report = {"status": "passed", "atol": 1e-3, "rtol": 1e-3, "minimum_cosine": 0.99999,
              "real_input_count": len(args.parity_input), "comparisons": comparisons}
    print("[parity] " + json.dumps(report))
    if args.parity_report:
        with open(args.parity_report, "w") as handle:
            json.dump(report, handle, indent=2)

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print(f"[onnx] ir={m.ir_version} opset={m.opset_import[0].version} ops={dict(sorted(ops.items()))}")


if __name__ == "__main__":
    main()
