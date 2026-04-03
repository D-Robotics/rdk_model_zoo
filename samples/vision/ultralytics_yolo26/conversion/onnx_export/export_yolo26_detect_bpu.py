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

"""YOLO26 Detection ONNX Export Script.

This script exports a YOLO26 detection model to a BPU-optimized ONNX format.
It modifies the `Detect` head to output raw feature maps in NHWC layout.

Usage:
    python3 export_yolo26_detect_bpu.py --weights yolo26n.pt --output yolo26n_det_bpu.onnx
"""

import os
import shutil
import argparse
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect


def main():
    """Main entry point for detection model export."""
    parser = argparse.ArgumentParser(description="YOLO26 Detect Export Script")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO26 .pt model')
    parser.add_argument('--output', type=str, default='yolo26_det_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    export_bpu_onnx(args.weights, args.output, args.imgsz)


def bpu_detect_forward(self, x):
    """Modified forward method for YOLO26 Detect Head (BPU-Optimized).

    Args:
        x (List[torch.Tensor]): Input feature maps from the neck.

    Returns:
        List[torch.Tensor]: A list of 6 tensors (for 3 scales): [Cls, Box] * 3.
        Layout is NHWC.
    """
    res = []
    
    # YOLO26 Logic: Prefer one2one (End-to-End) branch weights if available
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    for i in range(self.nl):
        # Cls Branch: NCHW -> NHWC
        scores = cls_layers[i](x[i]).permute(0, 2, 3, 1)
        # Box Branch: NCHW -> NHWC
        bboxes = box_layers[i](x[i]).permute(0, 2, 3, 1)
        
        res.append(scores)
        res.append(bboxes)
        
    return res


def export_bpu_onnx(model_path: str, output_name: str = "yolo26_bpu.onnx", imgsz: int = 640):
    """Export YOLO26 model to BPU-friendly ONNX."""
    print(f"Loading model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Applying BPU Monkey Patch (Output Layout: NHWC)...")
    Detect.forward = bpu_detect_forward
    
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
        if output_name and exported_path != output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            shutil.move(exported_path, output_name)
            print(f"Renamed output to: {output_name}")
            return output_name
        return exported_path
    else:
        print("❌ Export failed.")
        return None


if __name__ == "__main__":
    main()