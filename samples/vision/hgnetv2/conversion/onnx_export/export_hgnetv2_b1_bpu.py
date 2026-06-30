# Copyright (c) 2026 D-Robotics Corporation
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

"""Export the PP-HGNetV2 b1 backbone to a BPU-friendly ONNX file.

Run inside the OpenExplore RDK X5 toolchain Docker (torch 1.13). The script
loads pretrained weights via ``timm`` and writes ``hgnetv2_b1.onnx`` to the
current working directory, ready to be consumed by ``hb_mapper makertbin``.
"""

import timm
import torch

# 1. Load the pretrained model in eval mode.
model = timm.create_model("hgnetv2_b1.ssld_stage2_ft_in1k", pretrained=True)
model.eval()

# 2. Build a dummy input tensor (batch=1, RGB, 224x224) for tracing.
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export to ONNX (opset 11 matches the version supported by hb_mapper 1.24.3).
torch.onnx.export(
    model,
    dummy_input,
    "hgnetv2_b1.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
)

print("Model exported to hgnetv2_b1.onnx")
