import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Classify
import os
import shutil

# ==============================================================================
# 1. 定义适配 BPU 的 Classify Forward 函数
# ==============================================================================
def bpu_classify_forward(self, x):
    """
    YOLO26 Classify Head Modified for BPU (Fully Convolutional).
    
    Logic:
        Standard: x -> conv -> pool -> flatten -> drop -> linear -> softmax
        BPU Opt:  x -> conv -> pool -> drop -> conv1x1 (Replace Linear)
    
    Output:
        Tensor (B, NC, 1, 1) -> Effective (B, NC)
    """
    if isinstance(x, list):
        x = torch.cat(x, 1)
    
    # x: (B, C, H, W)
    x = self.conv(x)
    # pool: (B, C, 1, 1)
    x = self.pool(x)
    x = self.drop(x)
    
    # linear is now a Conv2d 1x1
    x = self.linear(x)
    
    # Explicitly permute to NHWC (B, 1, 1, C) for consistency
    x = x.permute(0, 2, 3, 1)
    
    return x

# ==============================================================================
# 2. 执行 Monkey Patch 并导出
# ==============================================================================
def export_cls_bpu(model_path, output_name="yolo26_cls_bpu.onnx", imgsz=224):
    print(f"Loading Classify model: {model_path}...")
    model = YOLO(model_path)
    
    # Convert Linear to Conv2d 1x1 for BPU efficiency
    # Locate the Classify Head directly (Standard Ultralytics structure)
    head = model.model.model[-1]
                
    if isinstance(head, Classify) and hasattr(head, 'linear') and isinstance(head.linear, nn.Linear):
        print("Converting Linear layer to Conv2d 1x1...")
        linear = head.linear
        in_ch = linear.in_features
        out_ch = linear.out_features
        
        # Create Conv2d
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        
        # Copy weights: Linear (out, in) -> Conv (out, in, 1, 1)
        conv.weight.data = linear.weight.data.view(out_ch, in_ch, 1, 1)
        conv.bias.data = linear.bias.data
        
        # Replace
        head.linear = conv
    
    # Monkey Patch
    Classify.forward = bpu_classify_forward
    print("Monkey patch applied (Fully Convolutional). Starting export...")
    
    # 执行导出
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False, 
        opset=11,
        simplify=True
    )
    
    if exported_path:
        # 重命名/移动
        if output_name:
            out_dir = os.path.dirname(output_name)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.move(exported_path, output_name)
            exported_path = output_name
            
        print(f"\n✅ Export success: {exported_path}")
        print("\n=== Output Node Description ===")
        print(f"Model has 1 output:")
        print(f"  Shape: (1, 1, 1, 1000) (NHWC)")
        print(f"  Content: Raw Logits (Needs Flatten & Softmax)")
        print("===============================")
    else:
        print("❌ Export failed.")
if __name__ == "__main__":
    # 使用 ImageNet 预训练模型
    # wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt
    MODEL_PATH = "yolo26n-cls.pt" 
    # BPU 命名习惯: model_platform_res_format
    OUTPUT_NAME = "yolo26n_cls_bayese_224x224_nv12.onnx"
    
    export_cls_bpu(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=224)
