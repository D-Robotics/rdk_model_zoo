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

def main():
    """Main entry point for segmentation model export."""
    parser = argparse.ArgumentParser(description='YOLO26 Segmentation Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26-seg .pt model')
    parser.add_argument('--output', type=str, default='yolo26_seg_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    args = parser.parse_args()
    export_seg_bpu(args.weights, args.output, args.imgsz, opset=args.opset if args.opset is not None else 11 if args.platform == 'x5' else 19, simplify=bool(args.simplify) if args.simplify is not None else args.platform == 'x5')

def bpu_segment_forward(self, x):
    """Modified forward method for YOLO26 Segment Head (BPU-Optimized).

    Returns:
        List[torch.Tensor]: 10 tensors ( [Cls, Box, MC] * 3 + Proto ).
        Layout is NHWC.
    """
    res = []
    if hasattr(self, 'one2one_cv2'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
        mc_layers = self.one2one_cv4
    else:
        box_layers = self.cv2
        cls_layers = self.cv3
        mc_layers = self.cv4
    for i in range(self.nl):
        feat = x[i]
        res.append(cls_layers[i](feat).permute(0, 2, 3, 1))
        res.append(box_layers[i](feat).permute(0, 2, 3, 1))
        res.append(mc_layers[i](feat).permute(0, 2, 3, 1))
    proto = self.proto(x)
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
