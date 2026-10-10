# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""YOLO26 Segmentation ONNX Export Script.

This script exports a YOLO26 segmentation model to a BPU-optimized ONNX format.
It modifies the `Segment` head to output raw tensors in NHWC layout.

Usage:
    python3 export_yolo26_seg_bpu.py --weights yolo26n-seg.pt --output yolo26n_seg_bpu.onnx
"""
import os
import shutil
import argparse
import sys

try:
    from batch_flex import adapt_calibration_batch8
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from batch_flex import adapt_calibration_batch8

def main():
    """Main entry point for segmentation model export."""
    parser = argparse.ArgumentParser(description='YOLO26 Segmentation Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26-seg .pt model')
    parser.add_argument('--output', type=str, default='yolo26_seg_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    parser.add_argument('--require-local', action='store_true',
                        help='fail unless --weights already exists locally')
    args = parser.parse_args()
    if args.require_local and not os.path.isfile(args.weights):
        parser.error(f'checkpoint does not exist locally: {args.weights}')
    export_seg_bpu(args.weights, args.output, args.imgsz, opset=args.opset if args.opset is not None else 11 if args.platform == 'x5' else 19, simplify=bool(args.simplify) if args.simplify is not None else args.platform == 'x5')

def bpu_segment_forward(self, x):
    """Modified forward method for YOLO26 Segment Head (BPU-Optimized).

    Returns:
        List[torch.Tensor]: 10 tensors ( [Cls, Box, MC] * 3 + Proto ).
        Layout is NHWC.
    """
    res = []
    one2one = tuple(getattr(self, name, None) for name in
                    ('one2one_cv2', 'one2one_cv3', 'one2one_cv4'))
    many = tuple(getattr(self, name, None) for name in ('cv2', 'cv3', 'cv4'))
    if all(layer is not None for layer in one2one):
        box_layers, cls_layers, mc_layers = one2one
    elif all(layer is not None for layer in many):
        box_layers, cls_layers, mc_layers = many
    else:
        raise RuntimeError('YOLO26 Segment export requires a complete detect and mask head')
    for i in range(self.nl):
        feat = x[i]
        res.append(cls_layers[i](feat).permute(0, 2, 3, 1))
        res.append(box_layers[i](feat).permute(0, 2, 3, 1))
        res.append(mc_layers[i](feat).permute(0, 2, 3, 1))
    head_types = {base.__name__ for base in type(self).__mro__}
    if 'Segment26' in head_types:
        proto = self.proto(x)
    elif 'Segment' in head_types:
        proto = self.proto(x[0])
    else:
        raise RuntimeError(f'unsupported segmentation head type: {type(self).__name__}')
    if isinstance(proto, (list, tuple)):
        proto = proto[0]
    res.append(proto.permute(0, 2, 3, 1))
    return res

def export_seg_bpu(model_path: str, output_name: str='yolo26_seg_bpu.onnx', imgsz: int=640, opset=11, simplify=True):
    """Export YOLO26 Segmentation model."""
    global torch, YOLO, Segment
    import torch
    try:
        from ultralytics.nn.modules import Segment26 as Segment
    except ImportError:
        from ultralytics.nn.modules import Segment
    from ultralytics import YOLO
    print(f'Loading Segment model: {model_path}...')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'Error loading model: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    print('Applying BPU Monkey Patch for Segmentation (Output Layout: NHWC)...')
    Segment.forward = bpu_segment_forward
    print(f'Starting export (imgsz={imgsz})...')
    try:
        exported_path = model.export(format='onnx', imgsz=imgsz, dynamic=False, opset=opset, simplify=simplify)
    except Exception as e:
        print(f'Export exception: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    if exported_path:
        adapt_calibration_batch8(exported_path, 'seg')
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
