#!/user/bin/env python

# Copyright (c) 2025，WuChao D-Robotics.
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

# 注意: 此程序在Ultralytics的环境中运行
# Attention: This program runs on Ultralytics environment.

from ultralytics import YOLO
from types import MethodType
import torch.nn as nn

# Initialize a YOLOE model
model = YOLO("yoloe-11s-seg-pf.pt")

# replace some module without retraining
def linear2conv(linear):
    assert isinstance(linear, nn.Linear), "Input must be a Linear layer."
    conv = nn.Conv2d(
        in_channels=linear.in_features, 
        out_channels=linear.out_features,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True if linear.bias is not None else False 
    )
    conv.weight.data = linear.weight.view(linear.out_features, linear.in_features, 1, 1).data
    conv.bias.data = linear.bias.data if linear.bias is not None else conv.bias.data
    return conv

def cauchy_rdk_forward(self, x, text): # RDK
    results = []
    for i in range(self.nl):
        results.append(self.lrpc[i].vocab(self.cv3[i](x[i])).permute(0, 2, 3, 1).contiguous())
        results.append(self.lrpc[i].loc(self.cv2[i](x[i])).permute(0, 2, 3, 1).contiguous())
        results.append(self.cv5[i](x[i]).permute(0, 2, 3, 1).contiguous())
    results.append(self.proto(x[0]).permute(0, 2, 3, 1).contiguous())
    return results

model.model.model[23].lrpc[0].vocab = linear2conv(model.model.model[23].lrpc[0].vocab)
model.model.model[23].lrpc[1].vocab = linear2conv(model.model.model[23].lrpc[1].vocab)
model.model.model[23].forward = MethodType(cauchy_rdk_forward, model.model.model[23])


# save names
names = ""
for value in model.names.values():
    print(value)
    names += f"{value}\n"

with open("model_names.txt", "w", encoding="utf-8") as file:
    file.write(names)
print("model_names.txt saved.")

# export
model.export(imgsz=640, format='onnx', simplify=True, opset=11)

