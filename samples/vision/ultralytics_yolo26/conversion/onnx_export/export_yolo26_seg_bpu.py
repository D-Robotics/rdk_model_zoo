# Copyright (c) 2025 D-Robotics Corporation
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

"""YOLO26 Segmentation ONNX Export Script.

This script exports a YOLO26 segmentation model to a BPU-optimized ONNX format.
It modifies the `Segment` head to output raw tensors in NHWC layout.

Usage:
    python3 export_yolo26_seg_bpu.py --weights yolo26n-seg.pt --output yolo26n_seg_bpu.onnx
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO
try:
    from ultralytics.nn.modules import Segment26 as Segment
except ImportError:
    from ultralytics.nn.modules import Segment


def main():
    """Main entry point for segmentation model export."""
    parser = argparse.ArgumentParser(description="YOLO26 Segmentation Export Script")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO26-seg .pt model')
    parser.add_argument('--output', type=str, default='yolo26_seg_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    export_seg_bpu(args.weights, args.output, args.imgsz)


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
        
    proto = self.proto(x[0] if isinstance(x, list) else x) 
    res.append(proto.permute(0, 2, 3, 1))
    
    return res


def export_seg_bpu(model_path: str, output_name: str = "yolo26_seg_bpu.onnx", imgsz: int = 640):
    """Export YOLO26 Segmentation model."""
    print(f"Loading Segment model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Applying BPU Monkey Patch for Segmentation (Output Layout: NHWC)...")
    Segment.forward = bpu_segment_forward
    
    print(f"Starting export (imgsz={imgsz})...")
    try:
        exported_path = model.export(
            format="onnx", imgsz=imgsz, dynamic=False, opset=11, simplify=True
        )
    except Exception as e:
        print(f"Export exception: {e}")
        return
    
    if exported_path:
        if output_name and exported_path != output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir: os.makedirs(out_dir, exist_ok=True)
            shutil.move(exported_path, output_name)
            exported_path = output_name
            
        print(f"\n✅ Export success: {exported_path}")
    else:
        print("❌ Export failed.")


if __name__ == "__main__":
    main()
