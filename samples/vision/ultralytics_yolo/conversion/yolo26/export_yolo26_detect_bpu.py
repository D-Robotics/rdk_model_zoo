# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""YOLO26 Detection ONNX Export Script.

This script exports a YOLO26 detection model to a BPU-optimized ONNX format.
It modifies the `Detect` head to output raw feature maps in NHWC layout.

Usage:
    python3 export_yolo26_detect_bpu.py --weights yolo26n.pt --output yolo26n_det_bpu.onnx
"""
import os
import shutil
import argparse

try:
    from workflow import export_defaults
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from workflow import export_defaults

def main():
    """Main entry point for detection model export."""
    parser = argparse.ArgumentParser(description='YOLO26 Detect Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26 .pt model')
    parser.add_argument('--output', type=str, default='yolo26_det_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    parser.add_argument('--require-local', action='store_true',
                        help='fail unless --weights already exists locally')
    args = parser.parse_args()
    if args.require_local and not os.path.isfile(args.weights):
        parser.error(f'checkpoint does not exist locally: {args.weights}')
    default_opset, default_simplify = export_defaults(args.platform, 'yolo26')
    export_bpu_onnx(
        args.weights, args.output, args.imgsz,
        opset=args.opset if args.opset is not None else default_opset,
        simplify=bool(args.simplify) if args.simplify is not None else default_simplify,
    )

def bpu_detect_forward(self, x):
    """Modified forward method for YOLO26 Detect Head (BPU-Optimized).

    Args:
        x (List[torch.Tensor]): Input feature maps from the neck.

    Returns:
        List[torch.Tensor]: A list of 6 tensors (for 3 scales): [Cls, Box] * 3.
        Layout is NHWC.
    """
    res = []
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3
    for i in range(self.nl):
        scores = cls_layers[i](x[i]).permute(0, 2, 3, 1)
        bboxes = box_layers[i](x[i]).permute(0, 2, 3, 1)
        res.append(scores)
        res.append(bboxes)
    return res

def export_bpu_onnx(model_path: str, output_name: str='yolo26_bpu.onnx', imgsz: int=640, opset=11, simplify=True):
    """Export YOLO26 model to BPU-friendly ONNX."""
    global torch, YOLO, Detect
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import Detect
    print(f'Loading model: {model_path}...')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'Error loading model: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    print('Applying BPU Monkey Patch (Output Layout: NHWC)...')
    Detect.forward = bpu_detect_forward
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
            print(f'Renamed output to: {output_name}')
            return output_name
        return exported_path
    else:
        raise RuntimeError('Exporter returned no ONNX artifact')
        return None
if __name__ == '__main__':
    main()
