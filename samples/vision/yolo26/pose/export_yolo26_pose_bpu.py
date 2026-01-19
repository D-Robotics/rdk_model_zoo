import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Pose
import os
import shutil

# ==============================================================================
# YOLO26 Pose 适配 BPU 的 Forward 函数
# ==============================================================================
def bpu_pose_forward(self, x):
    """
    YOLO26 Pose Head Modified for BPU.
    
    Outputs (NHWC Layout):
        List of Tensors (9 items for 3 scales):
        Scale 1 (P3): [Box, Cls, Kpts]
        Scale 2 (P4): [Box, Cls, Kpts]
        Scale 3 (P5): [Box, Cls, Kpts]
    """
    res = []
    
    # 优先使用 one2one (End-to-End) 分支
    # Pose 模型通常包含 one2one_cv2(box), one2one_cv3(cls), one2one_cv4(kpts/pose_feat)
    # YOLO26 Pose26 特有: one2one_cv4_kpts
    
    use_one2one = hasattr(self, 'one2one_cv2')
    if use_one2one:
        print("Exporting using YOLO26 End-to-End (One2One) branch weights.")
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
        # Pose26 check
        if hasattr(self, 'one2one_cv4_kpts'):
            pose_feat_layers = self.one2one_cv4
            kpts_head_layers = self.one2one_cv4_kpts
            is_pose26 = True
        else:
            kpt_layers = self.one2one_cv4
            is_pose26 = False
    else:
        print("Exporting using Standard branch weights.")
        box_layers = self.cv2
        cls_layers = self.cv3
        if hasattr(self, 'cv4_kpts'):
            pose_feat_layers = self.cv4
            kpts_head_layers = self.cv4_kpts
            is_pose26 = True
        else:
            kpt_layers = self.cv4
            is_pose26 = False

    for i in range(self.nl):
        feat = x[i] # Backbone/Neck Output
        
        # 1. Box 分支 (Regression) -> NHWC
        # Shape: (B, H, W, 4)
        bboxes = box_layers[i](feat).permute(0, 2, 3, 1)
        
        # 2. Cls 分支 (Classification) -> NHWC
        # Shape: (B, H, W, nc)
        scores = cls_layers[i](feat).permute(0, 2, 3, 1)
        
        # 3. Kpts 分支 (Keypoints) -> NHWC
        if is_pose26:
            # Pose26: Feat -> Head -> Permute
            pose_feat = pose_feat_layers[i](feat)
            kpts = kpts_head_layers[i](pose_feat).permute(0, 2, 3, 1)
        else:
            # Old Pose: Direct -> Permute
            kpts = kpt_layers[i](feat).permute(0, 2, 3, 1)
        
        res.append(bboxes)
        res.append(scores)
        res.append(kpts)
        
    return res

# ==============================================================================
# 导出主函数
# ==============================================================================
def export_pose_bpu(model_path, output_name="yolo26_pose_bpu.onnx", imgsz=640):
    print(f"Loading Pose model: {model_path}...")
    model = YOLO(model_path)
    
    # 强制替换 Pose 类的 forward 方法
    # 注意：YOLO Pose 模型 Head 是 Pose 类，继承自 Detect
    Pose.forward = bpu_pose_forward
    
    print("Monkey patch applied for POSE task. Starting export...")
    
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False, 
        opset=11,
        simplify=True
    )
    
    if exported_path:
        if output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.move(exported_path, output_name)
            exported_path = output_name
            
        print(f"\n✅ Export success: {exported_path}")
        print("\n=== Output Node Description ===")
        print("Model has 9 outputs (3 scales * 3 branches):")
        print("  Layout: NHWC")
        print("  0: P3 Box (1, 80, 80, 4)")
        print("  1: P3 Cls (1, 80, 80, nc)")
        print("  2: P3 Kpts (1, 80, 80, nk*3) <--- New!")
        print("  3: P4 Box ...")
        print("===============================")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    # 请确保您下载了 yolo26n-pose.pt
    # wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt
    MODEL_PATH = "yolo26n-pose.pt" 
    OUTPUT_NAME = "yolo26n_pose.onnx"
    
    export_pose_bpu(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=640)
