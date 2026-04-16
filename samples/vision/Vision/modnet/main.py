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


@dataclass
class MODNetConfig:
    model_path: str
    output_path: str
    ref_size: int = 512
    

class MODNet:
    def __init__(self, config: MODNetConfig):
        self.model = HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_H = self.input_shapes[self.input_names[0]][2]
        self.input_W = self.input_shapes[self.input_names[0]][3]
        self.cfg = config
        self.pad_x = None
        self.pad_y = None 
        self.new_h = None
        self.new_w = None
        self.orig_w = None
        self.orig_h = None
        
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
            
    def resize_with_padding(self, im, target_size=512):
        orig_h, orig_w, _ = im.shape

        scale = target_size / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_w = target_size - new_w
        pad_h = target_size - new_h

        pad_left = pad_w // 2
        pad_top = pad_h // 2

        im_padded = cv2.copyMakeBorder(
            im_resized,
            pad_top,
            pad_h - pad_top,
            pad_left,
            pad_w - pad_left,
            cv2.BORDER_CONSTANT,
            value=0,
        )

        return im_padded, scale, pad_left, pad_top, orig_w, orig_h, new_w, new_h
            
    def pre_process(self, im):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = (im - 127.5) / 127.5
        im_pad, scale, pad_x, pad_y, orig_w, orig_h, new_w, new_h = self.resize_with_padding(
            im, self.cfg.ref_size
        )
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.new_w = new_w
        self.new_h = new_h
        self.orig_h = orig_h
        self.orig_w = orig_w
        inp = np.transpose(im_pad, (2, 0, 1))
        inp = np.expand_dims(inp, 0).astype(np.float32)

        return {
            self.model_name: {
                self.input_names[0]: inp
            }
        }

    def predict(self, img):
        input_data = self.pre_process(img)
        outputs = self.model.run(input_data)
        return outputs[self.model_name][self.output_names[0]].squeeze()

    def post_process(self, matte):
        matte = (matte * 255).astype(np.uint8)
        matte_unpad = matte[self.pad_y : self.pad_y + self.new_h, self.pad_x : self.pad_x + self.new_w]
        matte_final = cv2.resize(
            matte_unpad, (self.orig_w, self.orig_h), interpolation=cv2.INTER_LINEAR
        )
        cv2.imwrite(self.cfg.output_path, matte_final)
    
    def __call__(self, img):
        return self.predict(img)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/root/rdk_model_zoo/demos/Vision/modnet/modnet_output.bin',
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--output-path', type=str,
                        default=f'matte.png')
    parser.add_argument('--priority', type=int, default=5,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='/root/modnet/cgz.jpg',
                        help='Path to load test image.')

    args = parser.parse_args()

    model_config = MODNetConfig(args.model_path, args.output_path)
    model = MODNet(model_config)
    model.set_scheduling_params(
        priority=args.priority,
        bpu_cores=args.bpu_cores
    )
    img = cv2.imread(args.test_img)
    res = model(img)

    model.post_process(res)
    

if __name__ == "__main__":
    main()