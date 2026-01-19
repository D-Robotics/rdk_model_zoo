import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Segment26, Detect
import os
import shutil

# ==============================================================================
# 1. 定义适配 BPU 的 Segment Forward 函数
# ==============================================================================
def bpu_segment_forward(self, x):
    """
    YOLO26 Segment Head Modified for BPU. 
    
    Outputs (NHWC Layout):
        List of Tensors (10 items):
        Scale 1-3: [Box, Cls, MC] * 3
        Last item: Proto (1, 160, 160, 32)
    """
    res = []
    
    # 1. 产生 Detection 和 MC 分支 (使用 One2One 权重)
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
        # Box -> NHWC
        bboxes = box_layers[i](feat).permute(0, 2, 3, 1)
        # Cls -> NHWC
        scores = cls_layers[i](feat).permute(0, 2, 3, 1)
        # Mask Coefficients -> NHWC
        mces = mc_layers[i](feat).permute(0, 2, 3, 1)
        
        res.append(bboxes)
        res.append(scores)
        res.append(mces)
        
    # 2. 产生 Proto 分支
    # YOLO26 Proto 通常接收所有尺度的特征 x
    proto = self.proto(x) 
    if isinstance(proto, (list, tuple)):
        proto = proto[0] # Handle multiple returns if any
        
    # Proto -> NHWC (1, 32, 160, 160) -> (1, 160, 160, 32)
    res.append(proto.permute(0, 2, 3, 1))
    
    return res

# ==============================================================================
# 2. 执行导出
# ==============================================================================
def export_seg_bpu(model_path, output_name="yolo26_seg_bpu.onnx", imgsz=640):
    print(f"Loading Segment model: {model_path}...")
    model = YOLO(model_path)
    
    # Monkey Patch
    # 注意: YOLO26 分割使用 Segment26 类
    Segment26.forward = bpu_segment_forward
    print("Monkey patch applied for SEG task. Starting export...")
    
    # 执行导出
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
        print("Model has 10 outputs:")
        print("  Layout: NHWC")
        print("  0-8: [Box, Cls, MC] for 3 scales")
        print("  9: Proto (1, 160, 160, 32)")
        print("===============================")
    else:
        print("❌ Export failed.")

if __name__ == "__main__":
    MODEL_PATH = "yolo26n-seg.pt" 
    OUTPUT_NAME = "yolo26n_seg_bayese_640x640_nv12.onnx"
    
    export_seg_bpu(MODEL_PATH, output_name=OUTPUT_NAME, imgsz=640)
