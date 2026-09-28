# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
"""YOLO26 Pose Estimation ONNX Export Script.

This script exports a YOLO26 pose estimation model to a BPU-optimized ONNX format.
It modifies the `Pose` head to output raw tensors in NHWC layout.

Usage:
    python3 export_yolo26_pose_bpu.py --weights yolo26n-pose.pt --output yolo26n_pose_bpu.onnx
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
    """Main entry point for pose model export."""
    parser = argparse.ArgumentParser(description='YOLO26 Pose Export Script')
    parser.add_argument('--weights', '--pt', type=str, required=True, help='Path to YOLO26-pose .pt model')
    parser.add_argument('--output', type=str, default='yolo26_pose_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--platform', choices=['x5', 's100', 's100p', 's600'], default='x5')
    parser.add_argument('--opset', '--optse', type=int, default=None)
    parser.add_argument('--simplify', type=int, choices=[0, 1], default=None)
    parser.add_argument('--require-local', action='store_true',
                        help='fail unless --weights already exists locally')
    args = parser.parse_args()
    if args.require_local and not os.path.isfile(args.weights):
        parser.error(f'checkpoint does not exist locally: {args.weights}')
    export_pose_bpu(args.weights, args.output, args.imgsz, opset=args.opset if args.opset is not None else 11 if args.platform == 'x5' else 19, simplify=bool(args.simplify) if args.simplify is not None else args.platform == 'x5')

def bpu_pose_forward(self, x):
    """Modified forward method for YOLO26 Pose Head (BPU-Optimized).

    Returns:
        List[torch.Tensor]: 9 tensors ( [Cls, Box, Kpts] * 3 scales ).
        Layout is NHWC.
    """
    res = []
    branch = None
    pose26 = hasattr(self, 'cv4_kpts') or hasattr(self, 'one2one_cv4_kpts')
    for prefix in ('one2one_', ''):
        box = getattr(self, prefix + 'cv2', None)
        cls = getattr(self, prefix + 'cv3', None)
        pose_feat = getattr(self, prefix + 'cv4', None)
        kpts_head = getattr(self, prefix + 'cv4_kpts', None)
        complete = (box is not None and cls is not None and pose_feat is not None
                    and (not pose26 or kpts_head is not None))
        if complete:
            branch = (box, cls, pose_feat, kpts_head)
            break
    if branch is None:
        raise RuntimeError('YOLO26 Pose export requires a complete detection and keypoint head')
    box_layers, cls_layers, pose_layers, kpts_head_layers = branch
    for i in range(self.nl):
        feat = x[i]
        res.append(cls_layers[i](feat).permute(0, 2, 3, 1))
        res.append(box_layers[i](feat).permute(0, 2, 3, 1))
        if pose26:
            pose_feat_layers = pose_layers
            kpts = kpts_head_layers[i](pose_feat_layers[i](feat)).permute(0, 2, 3, 1)
        else:
            kpts = pose_layers[i](feat).permute(0, 2, 3, 1)
        res.append(kpts)
    return res

def export_pose_bpu(model_path: str, output_name: str='yolo26_pose_bpu.onnx', imgsz: int=640, opset=11, simplify=True):
    """Export YOLO26 Pose model."""
    global YOLO, Pose
    from ultralytics import YOLO
    from ultralytics.nn.modules import Pose
    print(f'Loading Pose model: {model_path}...')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f'Error loading model: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    print('Applying BPU Monkey Patch for Pose (Output Layout: NHWC)...')
    Pose.forward = bpu_pose_forward
    print(f'Starting export (imgsz={imgsz})...')
    try:
        exported_path = model.export(format='onnx', imgsz=imgsz, dynamic=False, opset=opset, simplify=simplify)
    except Exception as e:
        print(f'Export exception: {e}')
        raise RuntimeError('YOLO26 export failed; see preceding error')
    if exported_path:
        adapt_calibration_batch8(exported_path, 'pose')
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
