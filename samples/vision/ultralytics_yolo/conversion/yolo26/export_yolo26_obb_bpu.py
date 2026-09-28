# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""YOLO26 Oriented Bounding Box (OBB) ONNX Export Script.

This script exports a YOLO26 OBB model to a BPU-optimized ONNX format.
It modifies the `OBB` head to output raw tensors in NHWC layout, providing separate
outputs for Box, Classification, and Angle at each scale.

Usage:
    python3 export_yolo26_obb_bpu.py --weights yolo26n-obb.pt --output yolo26n_obb_bpu.onnx
"""
import os
import shutil
import argparse
import types
import sys

try:
    from batch_flex import adapt_calibration_batch8
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from batch_flex import adapt_calibration_batch8

def main():
    """Main entry point for OBB export."""
    parser = argparse.ArgumentParser(description='YOLO26 OBB Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26-obb .pt model')
    parser.add_argument('--output', type=str, default='yolo26_obb_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    parser.add_argument('--require-local', action='store_true',
                        help='fail unless --weights already exists locally')
    args = parser.parse_args()
    if args.require_local and not os.path.isfile(args.weights):
        parser.error(f'checkpoint does not exist locally: {args.weights}')
    export_obb_bpu(args.weights, args.output, args.imgsz, opset=args.opset if args.opset is not None else 11 if args.platform == 'x5' else 19, simplify=bool(args.simplify) if args.simplify is not None else args.platform == 'x5')

def bpu_obb_forward(self, x):
    """Modified forward method for YOLO26 OBB Head (BPU-Optimized).

    Args:
        x (List[torch.Tensor]): Input feature maps from the neck.

    Returns:
        List[torch.Tensor]: A list of 9 tensors (for 3 scales):
            - [Cls, Box, Angle] * 3 scales
        All tensors are in NHWC layout.
    """
    res = []
    one2one = tuple(getattr(self, name, None) for name in
                    ('one2one_cv2', 'one2one_cv3', 'one2one_cv4'))
    many = tuple(getattr(self, name, None) for name in ('cv2', 'cv3', 'cv4'))
    if all(layer is not None for layer in one2one):
        box_layers, cls_layers, angle_layers = one2one
    elif all(layer is not None for layer in many):
        box_layers, cls_layers, angle_layers = many
    else:
        raise RuntimeError('YOLO26 OBB export requires complete box, class, and angle heads')
    for i in range(self.nl):
        feat = x[i]
        scores = cls_layers[i](feat).permute(0, 2, 3, 1)
        bboxes = box_layers[i](feat).permute(0, 2, 3, 1)
        angles = angle_layers[i](feat).permute(0, 2, 3, 1)
        res.append(scores)
        res.append(bboxes)
        res.append(angles)
    return res

def export_obb_bpu(model_path: str, output_name: str='yolo26_obb_bpu.onnx', imgsz: int=640, opset=11, simplify=True):
    """Export YOLO26 OBB model.

    Args:
        model_path (str): Path to input .pt file.
        output_name (str): Path to save output .onnx file.
        imgsz (int): Input image size.
    """
    global torch, YOLO, OBB, OBB26
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules import OBB, OBB26
    print(f'Loading OBB model: {model_path}...')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'Error loading model: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    head = model.model.model[-1]
    if isinstance(head, (OBB, OBB26)):
        print(f'Detected OBB Head: {type(head).__name__}')
        head.forward = types.MethodType(bpu_obb_forward, head)
        print('Monkey patch applied for OBB task (NHWC layout).')
    else:
        print(f'Error: Last layer is {type(head).__name__}, not OBB/OBB26.')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    print(f'Starting export (imgsz={imgsz})...')
    try:
        exported_path = model.export(format='onnx', imgsz=imgsz, dynamic=False, opset=opset, simplify=simplify)
    except Exception as e:
        print(f'Export exception: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    if exported_path:
        adapt_calibration_batch8(exported_path, 'obb')
        if output_name and exported_path != output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.move(exported_path, output_name)
            print(f'Renamed output to: {output_name}')
            exported_path = output_name
        print(f'\nExport success: {exported_path}')
        print('\n=== Output Node Description ===')
        print('Model has 9 outputs:')
        print('  Layout: NHWC')
        print('  0-8: [Cls(NC), Box(4), Angle(1)] per scale')
        print('===============================')
    else:
        raise RuntimeError('Exporter returned no ONNX artifact')
if __name__ == '__main__':
    main()
