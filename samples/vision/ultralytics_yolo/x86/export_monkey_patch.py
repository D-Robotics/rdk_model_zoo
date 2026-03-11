#!/usr/bin/env python

# Copyright (c) 2025, D-Robotics.
# Modified for parameter flexibility.

# 注意: 此程序在Ultralytics模型的训练环境中运行
# Attention: This program runs on Ultralytics training Environment.

import argparse
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect, v10Detect, Segment, OBB, Pose, Classify
from ultralytics.nn.modules.block import Attention, AAttn
import torch
import types
import os

def main():
    parser = argparse.ArgumentParser(description="YOLO to ONNX for Horizon BPU")
    # 常用参数
    parser.add_argument('--pt', type=str, default='./yolo11n.pt', help='path to *.pt model.')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version (default: 11).')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640, 640], help='image size as [h, w] or [size].')
    parser.add_argument('--simplify', action='store_true', help='whether to use onnxsim (usually False for BPU).')
    
    opt = parser.parse_args()

    # 1. 初始化 Ultralytics YOLO 模型
    if not os.path.exists(opt.pt):
        print(f"\033[1;31mError: {opt.pt} not found!\033[0m")
        return

    m = YOLO(opt.pt)

    # 2. 注入针对 BPU 优化的猴子补丁 (Monkey Patch)
    modelZooOptimizer(m.model.model)

    # 3. 导出到 ONNX
    print(f"\033[1;32m[Cauchy] Exporting model with opset={opt.opset}, imgsz={opt.imgsz}...\033[0m")
    
    m.export(
        format='onnx', 
        imgsz=opt.imgsz,
        opset=opt.opset, 
        simplify=opt.simplify,
        # 对于 BPU 部署，通常建议固定 Batch Size 为 1
        dynamic=False 
    )

def modelZooOptimizer(model):
    """
    递归遍历模型，替换 Head 层和 Attention 层为 BPU 友好版本。
    """
    for name, child in model.named_children():
        if type(child) == Classify:
            child.forward = types.MethodType(Classify_forward, child)
            print(f"\033[1;34m[Cauchy] Patched Classify: {name}\033[0m")
        elif type(child) == Detect:
            child.forward = types.MethodType(Detect_forward, child)
            print(f"\033[1;34m[Cauchy] Patched Detect: {name}\033[0m")
        elif type(child) == v10Detect:
            child.forward = types.MethodType(v10Detect_forward, child)
            print(f"\033[1;34m[Cauchy] Patched v10Detect: {name}\033[0m")
        elif type(child) == Segment:
            child.forward = types.MethodType(Segment_forward, child)
            print(f"\033[1;34m[Cauchy] Patched Segment: {name}\033[0m")
        elif type(child) == Pose:
            child.forward = types.MethodType(Pose_forward, child)
            print(f"\033[1;34m[Cauchy] Patched Pose: {name}\033[0m")
        elif type(child) == OBB:
            child.forward = types.MethodType(OBB_forward, child)
            print(f"\033[1;34m[Cauchy] Patched OBB: {name}\033[0m")
        elif type(child) == AAttn:
            child.forward = types.MethodType(AAttn_forward, child)
            print(f"\033[1;34m[Cauchy] Patched AAttn: {name}\033[0m")
        elif type(child) == Attention:
            child.forward = types.MethodType(Attention_forward, child)
            print(f"\033[1;34m[Cauchy] Patched Attention: {name}\033[0m")
        
        modelZooOptimizer(child)

# --- 以下为 BPU 优化的 Forward 函数实现 ---

def Attention_forward(self, x): 
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.permute(0, 3, 1, 2).contiguous() 
    max_attn = attn.max(dim=1, keepdim=True).values 
    exp_attn = torch.exp(attn - max_attn)
    sum_attn = exp_attn.sum(dim=1, keepdim=True)
    attn = exp_attn / sum_attn
    attn = attn.permute(0, 2, 3, 1).contiguous() 
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    x = self.proj(x)
    return x

def AAttn_forward(self, x):
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
        B, N, _ = qkv.shape
    q, k, v = (qkv.view(B, N, self.num_heads, self.head_dim * 3).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim, self.head_dim], dim=2))
    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.permute(0, 3, 1, 2).contiguous() 
    max_attn = attn.max(dim=1, keepdim=True).values 
    exp_attn = torch.exp(attn - max_attn)
    sum_attn = exp_attn.sum(dim=1, keepdim=True)
    attn = exp_attn / sum_attn
    attn = attn.permute(0, 2, 3, 1).contiguous() 
    x = v @ attn.transpose(-2, -1)
    x = x.permute(0, 3, 1, 2)
    v = v.permute(0, 3, 1, 2)
    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, C)
        v = v.reshape(B // self.area, N * self.area, C)
        B, N, _ = x.shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    x = x + self.pe(v)
    return self.proj(x)

def Classify_forward(self, x):
    x = torch.cat(x, 1) if isinstance(x, list) else x
    return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

def Detect_forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
    return result

def v10Detect_forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
    return result

def Segment_forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
    result.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous()) 
    return result

def Pose_forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
    return result

def OBB_forward(self, x):
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous()) 
    return result

if __name__ == '__main__':
    main()
