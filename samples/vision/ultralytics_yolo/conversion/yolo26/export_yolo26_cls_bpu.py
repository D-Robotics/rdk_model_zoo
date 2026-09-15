# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""YOLO26 Classification ONNX Export Script.

This script exports a YOLO26 classification model to a BPU-optimized ONNX format.
It replaces the final `Linear` layer with a `Conv2d 1x1` layer for BPU efficiency.

Usage:
    python3 export_yolo26_cls_bpu.py --weights yolo26n-cls.pt --output yolo26n_cls_bpu.onnx
"""
import os
import shutil
import argparse

def main():
    """Main entry point for classification model export."""
    parser = argparse.ArgumentParser(description='YOLO26 Classification Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26-cls .pt model')
    parser.add_argument('--output', type=str, default='yolo26_cls_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=224, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    args = parser.parse_args()
    export_cls_bpu(args.weights, args.output, args.imgsz, opset=args.opset if args.opset is not None else 11 if args.platform == 'x5' else 19, simplify=bool(args.simplify) if args.simplify is not None else args.platform == 'x5')

def bpu_classify_forward(self, x):
    """Modified forward method for YOLO26 Classify Head (Fully Convolutional)."""
    if isinstance(x, list):
        x = torch.cat(x, 1)
    x = self.conv(x)
    x = self.pool(x)
    x = self.drop(x)
    x = self.linear(x)
    return x.permute(0, 2, 3, 1)

def convert_linear_to_conv(model):
    """Replace the final Linear layer with a Conv2d 1x1 layer."""
    head = model.model.model[-1]
    if isinstance(head, Classify) and hasattr(head, 'linear') and isinstance(head.linear, nn.Linear):
        print('Converting Linear layer to Conv2d 1x1...')
        linear = head.linear
        in_ch, out_ch = (linear.in_features, linear.out_features)
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        conv.weight.data = linear.weight.data.view(out_ch, in_ch, 1, 1)
        conv.bias.data = linear.bias.data
        head.linear = conv
        return True
    return False

def export_cls_bpu(model_path: str, output_name: str='yolo26_cls_bpu.onnx', imgsz: int=224, opset=11, simplify=True):
    """Export YOLO26 Classification model."""
    global torch, nn, YOLO, Classify
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    from ultralytics.nn.modules import Classify
    print(f'Loading Classify model: {model_path}...')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'Error loading model: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    if convert_linear_to_conv(model):
        print('Linear layer conversion applied.')
    print('Applying BPU Monkey Patch (Output Layout: NHWC)...')
    Classify.forward = bpu_classify_forward
    print(f'Starting export (imgsz={imgsz})...')
    try:
        exported_path = model.export(format='onnx', imgsz=imgsz, dynamic=False, opset=opset, simplify=simplify)
    except Exception as e:
        print(f'Export exception: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    if exported_path:
        if output_name and exported_path != output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.move(exported_path, output_name)
            exported_path = output_name
        print(f'\nExport success: {exported_path}')
    else:
        raise RuntimeError('Exporter returned no ONNX artifact')
if __name__ == '__main__':
    main()
