"""Export one YOLO26 Depth model to either ONNX boundary of the mixed profile.

Boundaries:

- ``lite`` (variants ``l``/``x``): raw 192×192 depth logit; the board runtime
  applies ``clip → scale/bias → exp → resize`` on the CPU.
- ``log`` (variants ``n``/``s``/``m``): calibrated log-depth including the
  ``clip → scale/bias`` stage in-graph, ready for NV12 compilation.
"""

import argparse
import json
import shutil
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.block import AAttn, Attention
from ultralytics.nn.modules.head import Depth


def attention_forward(self, x):
    """Replace the standard attention forward path for X5 export.

    Args:
        self: Ultralytics attention module instance.
        x: Input feature tensor.

    Returns:
        Export-friendly attention output tensor.
    """
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
    output = output.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
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
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
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


def main():
    """Export one YOLO26 Depth weight file to ONNX."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--boundary", choices=("lite", "log"), default=None,
                        help="ONNX output boundary: lite (raw logit) or log (calibrated "
                             "log-depth). Default follows the mixed profile: lite for l/x, "
                             "log for n/s/m.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--opset", type=int, default=11)
    args = parser.parse_args()

    if args.boundary is None:
        args.boundary = "lite" if args.variant in ("l", "x") else "log"

    args.output_dir.mkdir(parents=True, exist_ok=False)
    copied_weights = args.output_dir / f"yolo26{args.variant}-depth-{args.boundary}.pt"
    shutil.copy2(args.weights, copied_weights)
    model = YOLO(str(copied_weights), task="depth")
    counts = patch_model(model.model.model, args.boundary)
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
    onnx_name = f"yolo26{args.variant}-depth_op{args.opset}_{args.boundary}.onnx"
    target = args.output_dir / onnx_name
    if exported.resolve() != target.resolve():
        exported.replace(target)
    if args.boundary == "lite":
        contract = {
            "name": "raw_logit",
            "shape": [1, args.imgsz // 4, args.imgsz // 4, 1],
            "layout": "NHWC",
            "cpu_postprocess": "log_depth=clip(raw_logit,-4,5)*cal_a+cal_b; depth=exp(log_depth); bilinear resize to source size",
        }
    else:
        contract = {
            "name": "log_depth",
            "shape": [1, args.imgsz // 4, args.imgsz // 4, 1],
            "layout": "NHWC",
            "cpu_postprocess": "depth=exp(log_depth); bilinear resize to source size",
        }
    report = {
        "variant": args.variant,
        "boundary": args.boundary,
        "imgsz": args.imgsz,
        "weights": copied_weights.name,
        "onnx": onnx_name,
        "patch_counts": counts,
        "output_contract": contract,
    }
    (args.output_dir / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
