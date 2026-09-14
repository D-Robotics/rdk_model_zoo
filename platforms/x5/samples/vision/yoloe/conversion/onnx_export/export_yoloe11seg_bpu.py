#!/usr/bin/env python3

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

"""Export YOLOE-11 Seg Prompt-Free model to ONNX for BPU deployment.

This script modifies the YOLOE model's vocabulary layers from Linear to
Conv2d (BPU-compatible) and patches the forward method to output 10 tensors
in NHWC layout. It also saves the 4585-class vocabulary to a .names file.

Usage:
    python3 export_yoloe11seg_bpu.py --weights yoloe-11s-seg-pf.pt --imgsz 640
"""

import argparse
import json
from pathlib import Path
import re
from types import MethodType

import torch
import torch.nn as nn


def linear2conv(linear: nn.Linear) -> nn.Conv2d:
    """Replace a Linear layer with an equivalent 1x1 Conv2d for BPU."""
    assert isinstance(linear, nn.Linear), "Input must be a Linear layer."
    conv = nn.Conv2d(
        in_channels=linear.in_features,
        out_channels=linear.out_features,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=linear.bias is not None,
    ).to(device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        conv.weight.copy_(linear.weight.reshape(linear.out_features, linear.in_features, 1, 1))
        if linear.bias is not None:
            conv.bias.copy_(linear.bias)
    conv.train(linear.training)
    return conv


def patch_head(network):
    """Accept a fused PF head, preserving existing convolutional vocab layers."""
    heads = [m for m in network.modules()
             if all(hasattr(m, key) for key in ("lrpc", "cv2", "cv3", "cv5", "proto", "nl"))]
    if len(heads) != 1:
        raise ValueError("Expected one fused YOLOE-11 PF segmentation head; prompt models are unsupported.")
    head = heads[0]
    if head.nl != 3 or len(head.lrpc) != 3 or head.reg_max != 16 or head.nm != 32:
        raise ValueError("Expected three scales, DFL=16 and 32 mask coefficients.")
    for branch in head.lrpc:
        if isinstance(branch.vocab, nn.Linear):
            branch.vocab = linear2conv(branch.vocab)
        elif not isinstance(branch.vocab, nn.Conv2d):
            raise TypeError(f"Unsupported PF vocabulary layer: {type(branch.vocab).__name__}")
    head.forward = MethodType(rdk_forward, head)
    return head


def rdk_forward(self, x, text=None):
    """Forward method patched for BPU output layout (NHWC, 10 tensors)."""
    results = []
    for i in range(self.nl):
        results.append(self.lrpc[i].vocab(self.cv3[i](x[i])).permute(0, 2, 3, 1).contiguous())
        results.append(self.lrpc[i].loc(self.cv2[i](x[i])).permute(0, 2, 3, 1).contiguous())
        results.append(self.cv5[i](x[i]).permute(0, 2, 3, 1).contiguous())
    results.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())
    return results


def main():
    parser = argparse.ArgumentParser(description="Export YOLOE-11 Seg PF to ONNX for BPU")
    parser.add_argument("--weights", type=str, default="yoloe-11s-seg-pf.pt",
                        help="Path to YOLOE pre-trained weights.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size for export.")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version.")
    parser.add_argument("--names-output", type=str, default=None,
                        help="Output path for the class names file.")
    args = parser.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file() or not re.fullmatch(r"yoloe-11[sml]-seg-pf\.pt", weights.name):
        parser.error("Provide a local yoloe-11{s,m,l}-seg-pf.pt checkpoint (no automatic download).")
    if args.imgsz != 640:
        parser.error("This X5 deployment contract currently supports --imgsz 640 only.")
    names_path = Path(args.names_output).resolve() if args.names_output else weights.with_suffix(".names")
    onnx_path = weights.with_suffix(".onnx")
    metadata_path = weights.with_suffix(".export.json")
    for path in (names_path, onnx_path, metadata_path):
        if path.exists():
            parser.error(f"Refusing to overwrite {path}; use a fresh export directory.")
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.model.cpu().float().eval()
    patch_head(model.model)
    names = model.names
    if not isinstance(names, dict) or set(names) != set(range(4585)):
        raise ValueError("Expected 4585 contiguous class IDs in the PF checkpoint.")

    exported = model.export(imgsz=args.imgsz, format="onnx", simplify=True,
                            opset=args.opset, batch=1, dynamic=False, device="cpu")
    if Path(exported).resolve() != onnx_path or not onnx_path.is_file():
        raise RuntimeError(f"Unexpected export path: {exported}; expected {onnx_path}")
    with names_path.open("x", encoding="utf-8") as f:
        f.write("".join(f"{names[i]}\n" for i in range(4585)))
    import ultralytics
    with metadata_path.open("x", encoding="utf-8") as f:
        json.dump({"weights": str(weights), "onnx": str(onnx_path),
                   "names": str(names_path), "ultralytics": ultralytics.__version__,
                   "torch": torch.__version__, "opset": args.opset,
                   "validation": "exported_only_not_accuracy_verified"}, f, indent=2)
    print("Export complete.")


if __name__ == "__main__":
    main()
