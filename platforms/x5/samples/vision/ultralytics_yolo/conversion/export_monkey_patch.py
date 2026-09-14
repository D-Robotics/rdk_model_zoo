#!/user/bin/env python

# Copyright (c) 2025, WuChao D-Robotics.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', type=str, default='./yolo11n.pt', help='path to *.pt model.')
    parser.add_argument('--optse', type=int, default=11, help='opset version.')
    opt = parser.parse_args()

    # Init Ultralytics YOLO Model
    m = YOLO(opt.pt)

    # Replace some efficient modules
    modelZooOptimizer(m.model.model)

    # Export to ONNX
    m.export(format='onnx', simplify=False, opset=11)


def modelZooOptimizer(model):  # Monkey Patch
    for name, child in model.named_children():
        # print(name)
        if type(child) == Classify:
            child.forward = types.MethodType(Classify_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced Classify_forward in {name}" + "\033[0m")
        elif type(child) == Detect:
            child.forward = types.MethodType(Detect_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced Detect_forward in {name}" + "\033[0m")
        elif type(child) == v10Detect:
            child.forward = types.MethodType(v10Detect_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced v10Detect_forward in {name}" + "\033[0m")
        elif type(child) == Segment:
            child.forward = types.MethodType(Segment_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced Segment_forward in {name}" + "\033[0m")
        elif type(child) == Pose:
            child.forward = types.MethodType(Pose_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced Pose_forward in {name}" + "\033[0m")
        elif type(child) == OBB:
            child.forward = types.MethodType(OBB_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced OBB_forward in {name}" + "\033[0m")
        elif type(child) == AAttn:
            child.forward = types.MethodType(AAttn_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced AAttn_forward in {name}" + "\033[0m")
        elif type(child) == Attention:
            child.forward = types.MethodType(Attention_forward, child)
            print("\033[1;31m" + f"[Cauchy] Replaced Attention_forward in {name}" + "\033[0m")
        modelZooOptimizer(child)



def Attention_forward(self, x): 
    # Effieicient for Bayes-e BPU
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
    max_attn = attn.max(dim=1, keepdim=True).values 
    exp_attn = torch.exp(attn - max_attn)
    sum_attn = exp_attn.sum(dim=1, keepdim=True)
    attn = exp_attn / sum_attn
    attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
    x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
    x = self.proj(x)
    return x

def AAttn_forward(self, x):
    # Effieicient for Bayes-e BPU
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
        B, N, _ = qkv.shape
    q, k, v = (qkv.view(B, N, self.num_heads, self.head_dim * 3).permute(0, 2, 3, 1).split([self.head_dim, self.head_dim, self.head_dim], dim=2))
    attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
    attn = attn.permute(0, 3, 1, 2).contiguous()  # CHW2HWC like
    max_attn = attn.max(dim=1, keepdim=True).values 
    exp_attn = torch.exp(attn - max_attn)
    sum_attn = exp_attn.sum(dim=1, keepdim=True)
    attn = exp_attn / sum_attn
    attn = attn.permute(0, 2, 3, 1).contiguous()  # HWC2CHW like
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
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    x = torch.cat(x, 1) if isinstance(x, list) else x
    return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

def Detect_forward(self, x):
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
    return result
def v10Detect_forward(self, x):
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    result = []
    for i in range(self.nl):
        result.append(self.one2one_cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        result.append(self.one2one_cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
    return result

def Segment_forward(self, x):
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # proto weights
    result.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())       # proto mask
    return result

def Pose_forward(self, x):
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # kpts
    return result

def OBB_forward(self, x):
    # Effieicient for Bernoulli2, Bayes, Bayes-e, Nash-{e/m/p} BPU
    # TODO: Test and PostProcess Code in Model Zoo.
    result = []
    for i in range(self.nl):
        result.append(self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous())  # cls
        result.append(self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous())  # bbox
        result.append(self.cv4[i](x[i]).permute(0, 2, 3, 1).contiguous())  # theta logits
    return result


if __name__ == '__main__':
    main()