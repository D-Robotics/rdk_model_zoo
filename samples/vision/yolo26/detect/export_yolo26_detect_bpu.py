import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
import os
import shutil

# ==============================================================================
# 1. 定义适配 BPU 的 Forward 函数
# ==============================================================================
def bpu_detect_forward(self, x):
    """
    YOLO26 Detect Head Modified for BPU.
    
    Output:
        List of Tensors (6 items for 3 scales), Layout: NHWC
        [
          Scale1_Box_Raw (B, 80, 80, 64),  <-- 16 * 4 = 64 (reg_max=16) or 4
          Scale1_Cls_Raw (B, 80, 80, 80),  <-- nc=80
          ...
        ]
    """
    res = []
    
    # YOLO26 核心逻辑：优先使用 one2one (End-to-End) 分支的权重
    if hasattr(self, 'one2one_cv2') and hasattr(self, 'one2one_cv3'):
        box_layers = self.one2one_cv2
        cls_layers = self.one2one_cv3
    else:
        box_layers = self.cv2
        cls_layers = self.cv3

    for i in range(self.nl):
        
        # 1. Box 分支
        # NHWC: permute(0, 2, 3, 1)
        bboxes = box_layers[i](x[i]).permute(0, 2, 3, 1)
        
        # 2. Cls 分支
        # NHWC
        scores = cls_layers[i](x[i]).permute(0, 2, 3, 1)
        
        res.append(bboxes)
        res.append(scores)
        
    return res

# ==============================================================================
# 2. 执行 Monkey Patch 并导出
# ==============================================================================
def export_bpu_onnx(model_path, output_name="yolo26_bpu.onnx", imgsz=640):
    # 加载模型
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)
    
    # 强制替换 Detect 类的 forward 方法
    Detect.forward = bpu_detect_forward
    print("Monkey patch applied (Output Layout: NHWC). Starting export...")
    
    # 执行导出
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False, 
        opset=11,
        simplify=True
    )
    
    if exported_path:
        print(f"\n✅ Export success: {exported_path}")
        
        # 重命名/移动到指定输出名称
        if output_name:
            # 确保目录存在
            out_dir = os.path.dirname(output_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            shutil.move(exported_path, output_name)
            print(f"Renamed to: {output_name}")
            exported_path = output_name
            
        return exported_path
    else:
        print("❌ Export failed.")
        return None

if __name__ == "__main__":
    MODEL_PATH = "yolo26n.pt" 
    OUTPUT_NAME = "yolo26n_bpu.onnx"

    export_bpu_onnx(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=640)