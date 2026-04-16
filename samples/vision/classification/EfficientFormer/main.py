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


import os
import sys
import json
import numpy as np
from hbm_runtime import HB_HBMRuntime
import cv2
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Literal
import argparse

sys.path.append(os.path.abspath("../../.."))
import utils.py_utils.preprocess_utils as preprocess_utils


def np_softmax(arr):
    exp_out = np.exp(arr)
    sum_exp_out = np.sum(exp_out, axis=-1, keepdims=True)
    probs = exp_out / sum_exp_out
    return probs


def sample_logits(logits, fixed_output_token, temperature=1.0, top_p=0.8):
    probs = np_softmax(logits)
    max_index = np.argmax(probs)
    return max_index, probs


@dataclass
class EfficientFormerConfig:
    model_path: str
    

class EfficientFormer:
    def __init__(self, config: EfficientFormerConfig):
        self.model = HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        self.cfg = config
        
    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)
            
    def pre_process(self, img, resize_type: int = 1):
        resize_img = preprocess_utils.resized_image(img, self.input_W, self.input_H, resize_type)
        y, uv = preprocess_utils.bgr_to_nv12_planes(resize_img)
        y = y.astype(np.uint8)
        uv = uv.astype(np.uint8)

        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))

        return {
            self.model_name: {
                self.input_names[0]: nv12
            }
        }

    def predict(self, img):
        input_data = self.pre_process(img)
        outputs = self.model.run(input_data)
        return outputs[self.model_name][self.output_names[0]].squeeze()

    def post_process(self, logits):
        max_ind, probs = sample_logits(logits, True) 
        with open('data/ImageNet_1k.json', 'r', encoding='utf-8') as json_file:
            labels = json.load(json_file)
        print(labels[max_ind], probs[max_ind])
    
    def __call__(self, img):
        return self.predict(img)
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'model/EfficientFormer_l3_224x224_nv12.bin',
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--priority', type=int, default=5,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='data/bittern.JPEG',
                        help='Path to load test image.')

    args = parser.parse_args()

    model_config = EfficientFormerConfig(args.model_path)
    model = EfficientFormer(model_config)
    model.set_scheduling_params(
        priority=args.priority,
        bpu_cores=args.bpu_cores
    )
    img = cv2.imread(args.test_img)
    res = model(img)

    model.post_process(res)
    

if __name__ == "__main__":
    main()