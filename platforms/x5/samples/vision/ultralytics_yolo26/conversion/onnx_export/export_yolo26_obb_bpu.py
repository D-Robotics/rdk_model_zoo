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
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import OBB, OBB26


def main():
    """Main entry point for OBB export."""
    parser = argparse.ArgumentParser(description="YOLO26 OBB Export Script")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO26-obb .pt model')
    parser.add_argument('--output', type=str, default='yolo26_obb_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    export_obb_bpu(args.weights, args.output, args.imgsz)


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
    
    # YOLO26 Logic: Prefer one2one (End-to-End) branch weights if available
    if hasattr(self, 'one2one_cv2'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
        angle_layers = self.one2one_cv4
    else:
        box_layers = self.cv2
        cls_layers = self.cv3
        angle_layers = self.cv4

    for i in range(self.nl):
        feat = x[i]
        # 1. Cls Branch: NCHW -> NHWC
        scores = cls_layers[i](feat).permute(0, 2, 3, 1)
        # 2. Box Branch: NCHW -> NHWC
        bboxes = box_layers[i](feat).permute(0, 2, 3, 1)
        # 3. Angle Branch: NCHW -> NHWC
        angles = angle_layers[i](feat).permute(0, 2, 3, 1)
        
        res.append(scores)
        res.append(bboxes)
        res.append(angles)
        
    return res


def export_obb_bpu(model_path: str, output_name: str = "yolo26_obb_bpu.onnx", imgsz: int = 640):
    """Export YOLO26 OBB model.

    Args:
        model_path (str): Path to input .pt file.
        output_name (str): Path to save output .onnx file.
        imgsz (int): Input image size.
    """
    print(f"Loading OBB model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Locate Head
    head = model.model.model[-1]
    if isinstance(head, (OBB, OBB26)):
        print(f"Detected OBB Head: {type(head).__name__}")
        # Apply Monkey Patch using types.MethodType for safer binding
        head.forward = types.MethodType(bpu_obb_forward, head)
        print("Monkey patch applied for OBB task (NHWC layout).")
    else:
        print(f"Error: Last layer is {type(head).__name__}, not OBB/OBB26.")
        return

    # Export
    print(f"Starting export (imgsz={imgsz})...")
    try:
        exported_path = model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=False, 
            opset=11,
            simplify=True
        )
    except Exception as e:
        print(f"Export exception: {e}")
        return
    
    if exported_path:
        # Rename/Move
        if output_name and exported_path != output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.move(exported_path, output_name)
            print(f"Renamed output to: {output_name}")
            exported_path = output_name
            
        print(f"\n✅ Export success: {exported_path}")
        print("\n=== Output Node Description ===")
        print("Model has 9 outputs:")
        print("  Layout: NHWC")
        print("  0-8: [Cls(NC), Box(4), Angle(1)] per scale")
        print("===============================")
    else:
        print("❌ Export failed.")


if __name__ == "__main__":
    main()