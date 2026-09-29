# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export target-specific depth boundaries while preserving source graph patches.

All X5 variants and S n/s/m export calibrated log-depth (clip + scale/bias).
S l/x export raw logits; their CPU decoder applies calibration and exp.
Explicit alternate S boundaries are experiments, not published runtime profiles.
Torch and Ultralytics are imported only after CLI prerequisite validation.
"""

import argparse
import json
import shutil
import types
from pathlib import Path


def attention_forward(self, x):
    """Replace the standard attention forward path for X5 export.

    Args:
        self: Ultralytics attention module instance.
        x: Input feature tensor.

    Returns:
        Export-friendly attention output tensor.
    """
    batch, channels, height, width = x.shape
    tokens = height * width
    qkv = self.qkv(x)
    q, k, v = qkv.view(
        batch, self.num_heads, self.key_dim * 2 + self.head_dim, tokens
    ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    attention = (q.transpose(-2, -1) @ k) * self.scale
    attention = attention.permute(0, 3, 1, 2).contiguous()
    maximum = attention.max(dim=1, keepdim=True).values
    exponent = torch.exp(attention - maximum)
    attention = exponent / exponent.sum(dim=1, keepdim=True)
    attention = attention.permute(0, 2, 3, 1).contiguous()
    output = (v @ attention.transpose(-2, -1)).view(batch, channels, height, width)
    output = output + self.pe(v.reshape(batch, channels, height, width))
    return self.proj(output)


def area_attention_forward(self, x):
    """Replace area-attention execution with an export-friendly graph.

    Args:
        self: Ultralytics area-attention module instance.
        x: Input feature tensor.

    Returns:
        Export-friendly area-attention output tensor.
    """
    batch, channels, height, width = x.shape
    tokens = height * width
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(batch * self.area, tokens // self.area, channels * 3)
        batch, tokens, _ = qkv.shape
    q, k, v = (
        qkv.view(batch, tokens, self.num_heads, self.head_dim * 3)
        .permute(0, 2, 3, 1)
        .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
    )
    attention = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attention = attention.permute(0, 3, 1, 2).contiguous()
    maximum = attention.max(dim=1, keepdim=True).values
    exponent = torch.exp(attention - maximum)
    attention = exponent / exponent.sum(dim=1, keepdim=True)
    attention = attention.permute(0, 2, 3, 1).contiguous()
    output = (v @ attention.transpose(-2, -1)).permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    if self.area > 1:
        output = output.reshape(batch // self.area, tokens * self.area, channels)
        v = v.reshape(batch // self.area, tokens * self.area, channels)
        batch, tokens, _ = output.shape
    output = (
        output.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
    )
    v = v.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
    return self.proj(output + self.pe(v))


def depth_lite_forward(self, features):
    """Return the raw 192×192 depth logit for external lite postprocessing.

    Args:
        self: Ultralytics depth-head module instance.
        features: Multi-scale feature tensors from the backbone.

    Returns:
        Raw depth-logit tensor in NHWC layout.
    """
    projected = [self.proj[index](features[index]) for index in range(self.nl)]
    output = projected[-1]
    for index in range(self.nl - 2, -1, -1):
        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )
        output = self.refine[index](output + projected[index])
    raw = self.head(output)
    return raw.permute(0, 2, 3, 1).contiguous()


def depth_log_forward(self, features):
    """Return calibrated log-depth (clip + scale/bias) for the NV12 profile.

    Args:
        self: Ultralytics depth-head module instance.
        features: Multi-scale feature tensors from the backbone.

    Returns:
        Calibrated log-depth tensor in NHWC layout.
    """
    raw = depth_lite_forward(self, features)
    return raw.clamp(-4.0, 5.0) * self.cal_a + self.cal_b


def patch_model(module, boundary):
    """Patch supported Ultralytics modules for ONNX export.

    Args:
        module: Root PyTorch module to inspect and patch in place.
        boundary: ``lite`` (raw logit) or ``log`` (calibrated log-depth).

    Returns:
        The patch counts per module type.
    """
    from ultralytics.nn.modules.block import AAttn, Attention
    from ultralytics.nn.modules.head import Depth

    depth_forward = depth_lite_forward if boundary == "lite" else depth_log_forward
    counts = {"Depth": 0, "Attention": 0, "AAttn": 0}
    for child in module.modules():
        if type(child) is Depth:
            child.forward = types.MethodType(depth_forward, child)
            counts["Depth"] += 1
        elif type(child) is Attention:
            child.forward = types.MethodType(attention_forward, child)
            counts["Attention"] += 1
        elif type(child) is AAttn:
            child.forward = types.MethodType(area_attention_forward, child)
            counts["AAttn"] += 1
    return counts


def resolve_boundary(target, variant, requested):
    if target not in ("x5", "s100", "s100p", "s600") or variant not in (
        "n",
        "s",
        "m",
        "l",
        "x",
    ):
        raise ValueError("Unknown target/variant")
    boundary = requested or (
        "lite" if target != "x5" and variant in ("l", "x") else "log"
    )
    if boundary not in ("log", "lite") or (target == "x5" and boundary != "log"):
        raise ValueError(
            "X5 supports only log boundary; unknown boundaries are refused"
        )
    return boundary


def export_name(variant, opset, boundary):
    return f"yolo26{variant}-depth_op{opset}_{boundary}.onnx"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export depth ONNX; Torch/Ultralytics load only after argument validation"
    )
    parser.add_argument(
        "--target", required=True, choices=("x5", "s100", "s100p", "s600")
    )
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("n", "s", "m", "l", "x"))
    parser.add_argument(
        "--boundary",
        choices=("lite", "log"),
        help="Default follows target/variant; alternate S profiles are experiments",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, choices=(768,), default=768)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args(argv)
    boundary = resolve_boundary(args.target, args.variant, args.boundary)
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.opset <= 0:
        raise ValueError("opset must be positive")
    global torch, F
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    args.output_dir.mkdir(parents=True, exist_ok=False)
    copied_weights = args.output_dir / f"yolo26{args.variant}-depth-{boundary}.pt"
    shutil.copy2(args.weights, copied_weights)
    model = YOLO(str(copied_weights), task="depth")
    counts = patch_model(model.model.model, boundary)
    if counts["Depth"] != 1:
        raise ValueError(f'Expected exactly one Depth head, found {counts["Depth"]}')
    from ultralytics.nn.modules.head import Depth

    head = next(m for m in model.model.model.modules() if type(m) is Depth)
    calibration = {
        "cal_a": float(head.cal_a),
        "cal_b": float(head.cal_b),
        "clip": [-4, 5],
    }
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=False,
            dynamic=False,
            batch=1,
            device="cpu",
            half=False,
        )
    )
    target = args.output_dir / export_name(args.variant, args.opset, boundary)
    if exported.resolve() != target.resolve():
        exported.replace(target)
    report = {
        "target": args.target,
        "variant": args.variant,
        "boundary": boundary,
        "imgsz": args.imgsz,
        "onnx": target.name,
        "weights": copied_weights.name,
        "patch_counts": counts,
        "checkpoint_calibration": calibration,
        "output_contract": {
            "shape": [1, 192, 192, 1],
            "layout": "NHWC",
            "name": "raw_logit" if boundary == "lite" else "calibrated_log_depth",
        },
        "runtime_requirement": "For lite, checkpoint calibration must match runtime binding; custom weights need an explicit new binding",
        "compilation": "not-run",
    }
    (args.output_dir / "export-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
