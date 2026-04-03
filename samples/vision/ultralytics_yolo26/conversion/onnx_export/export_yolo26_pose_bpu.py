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

"""YOLO26 Pose Estimation ONNX Export Script.

This script exports a YOLO26 pose estimation model to a BPU-optimized ONNX format.
It modifies the `Pose` head to output raw tensors in NHWC layout.

Usage:
    python3 export_yolo26_pose_bpu.py --weights yolo26n-pose.pt --output yolo26n_pose_bpu.onnx
"""

import os
import shutil
import argparse
from ultralytics import YOLO
from ultralytics.nn.modules import Pose


def main():
    """Main entry point for pose model export."""
    parser = argparse.ArgumentParser(description="YOLO26 Pose Export Script")
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO26-pose .pt model')
    parser.add_argument('--output', type=str, default='yolo26_pose_bpu.onnx', help='Output ONNX path')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    
    args = parser.parse_args()
    
    export_pose_bpu(args.weights, args.output, args.imgsz)


def bpu_pose_forward(self, x):
    """Modified forward method for YOLO26 Pose Head (BPU-Optimized).

    Returns:
        List[torch.Tensor]: 9 tensors ( [Cls, Box, Kpts] * 3 scales ).
        Layout is NHWC.
    """
    res = []
    use_one2one = hasattr(self, 'one2one_cv2')
    
    if use_one2one:
        box_layers, cls_layers = self.one2one_cv2, self.one2one_cv3
        if hasattr(self, 'one2one_cv4_kpts'):
            pose_feat_layers, kpts_head_layers, is_pose26 = self.one2one_cv4, self.one2one_cv4_kpts, True
        else:
            kpt_layers, is_pose26 = self.one2one_cv4, False
    else:
        box_layers, cls_layers = self.cv2, self.cv3
        if hasattr(self, 'cv4_kpts'):
            pose_feat_layers, kpts_head_layers, is_pose26 = self.cv4, self.cv4_kpts, True
        else:
            kpt_layers, is_pose26 = self.cv4, False

    for i in range(self.nl):
        feat = x[i]
        res.append(cls_layers[i](feat).permute(0, 2, 3, 1))
        res.append(box_layers[i](feat).permute(0, 2, 3, 1))
        if is_pose26:
            kpts = kpts_head_layers[i](pose_feat_layers[i](feat)).permute(0, 2, 3, 1)
        else:
            kpts = kpt_layers[i](feat).permute(0, 2, 3, 1)
        res.append(kpts)
        
    return res


def export_pose_bpu(model_path: str, output_name: str = "yolo26_pose_bpu.onnx", imgsz: int = 640):
    """Export YOLO26 Pose model."""
    print(f"Loading Pose model: {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e: 
        print(f"Error loading model: {e}"); return

    print("Applying BPU Monkey Patch for Pose (Output Layout: NHWC)...")
    Pose.forward = bpu_pose_forward
    
    print(f"Starting export (imgsz={imgsz})...")
    try:
        exported_path = model.export(
            format="onnx", imgsz=imgsz, dynamic=False, opset=11, simplify=True
        )
    except Exception as e:
        print(f"Export exception: {e}"); return
    
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
