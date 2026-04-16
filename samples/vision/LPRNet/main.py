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


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]


def reprocess(pred_data):
    pred_label = np.argmax(pred_data, axis=0)
    no_repeat_blank_label = []
    pre_c = pred_label[0]
    if pre_c != len(CHARS) - 1: 
        no_repeat_blank_label.append(pre_c)
    for c in pred_label: 
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    char_list = [CHARS[i] for i in no_repeat_blank_label]
    return ''.join(char_list)


@dataclass
class LPRNetConfig:
    model_path: str
    test_img: str
    

class LPRNet:
    def __init__(self, config: LPRNetConfig):
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
            
    def pre_process(self):
        input_data = np.fromfile(self.cfg.test_img, dtype=np.float32).reshape(1,3,24,94)

        return {
            self.model_name: {
                self.input_names[0]: input_data
            }
        }

    def predict(self):
        input_data = self.pre_process()
        outputs = self.model.run(input_data)
        return outputs[self.model_name][self.output_names[0]].squeeze()

    def post_process(self, logits):
        plate_str = reprocess(logits)
        print(plate_str)

    def __call__(self):
        return self.predict()
    
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/root/rdk_model_zoo/demos/detect/LPRNet/lpr.bin',
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--priority', type=int, default=5,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='/root/rdk_model_zoo/demos/detect/LPRNet/test.bin',
                        help='Path to load test image.')

    args = parser.parse_args()

    model_config = LPRNetConfig(args.model_path, args.test_img)
    model = LPRNet(model_config)
    model.set_scheduling_params(
        priority=args.priority,
        bpu_cores=args.bpu_cores
    )
    res = model()
    model.post_process(res)
    

if __name__ == "__main__":
    main()