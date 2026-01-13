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

# flake8: noqa: E501
# flake8: noqa: E402

import os
import sys
import argparse
import hbm_runtime
import numpy as np
from typing import Optional, Dict

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../../.."))
import utils.py_utils.preprocess_utils as pre_utils
import utils.py_utils.postprocess_utils as post_utils
import utils.py_utils.common_utils as common


class MobileNetV2:
    """
    @brief MobileNetV2 classification model wrapper using HB_HBMRuntime backend.

    This class handles model loading, input preprocessing, inference, and postprocessing
    for running image classification tasks.
    """

    def __init__(self, opt):
        """
        @brief Initialize the MobileNetV2 model runtime and input/output configurations.

        @param opt (argparse.Namespace) Argument object with `model_path` key.
        """
        # Initialize model from path
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Get model name and input/output info
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.shapes = self.model.input_shapes[self.model_name]

        # Extract input resolution
        self.input_H = self.shapes[self.input_names[0]][2]
        self.input_W = self.shapes[self.input_names[0]][3]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set inference scheduling parameters like BPU core assignment and priority.

        @param priority (int, optional) Inference priority from 0 to 255.
        @param bpu_cores (list[int], optional) List of BPU core indices.
        @return None
        """
        kwargs = {}

        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self, img , resize_type: int = 1):
        """
        @brief 使用第一个代码的前处理方法
        """
        resize_img = pre_utils.resized_image(img, self.input_W, self.input_H, resize_type)
        y, uv = pre_utils.bgr_to_nv12_planes(resize_img)
        y = y.astype(np.uint8)
        uv = uv.astype(np.uint8)

        nv12 = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0)
        nv12 = nv12.reshape((1, self.input_H * 3 // 2, self.input_W, 1))

        return {
            self.model_name: {
                self.input_names[0]: nv12
            }
        }


    def forward(self,
                input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        @brief Run forward inference with the preprocessed input.

        @param input_tensor (dict) Nested input tensor dictionary.
        @return dict: Output tensor dictionary indexed by output names.
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     idx2label: Dict[int, str],
                     topk: int = 5) -> None:
        """
        @brief Postprocess classification output and print top-K results.

        @param outputs (dict) Output tensor dictionary from inference.
        @param idx2label (dict) Mapping from class index to label name.
        @param topk Number of top predictions to display.
        @return None
        """
        # Use utility function to print top-K predictions
        prob = np.squeeze(outputs[self.output_names[0]])

        topk_idx = np.argsort(prob)[-topk:][::-1]
        topk_prob = prob[topk_idx]

        # print result
        print(f"Top-{topk} Predictions:")
        for i in range(topk):
            idx = topk_idx[i]
            score = topk_prob[i]
            label = idx2label[int(idx)] if idx2label and int(idx) in idx2label else f"Class {int(idx)}"
            print(f"{label}: {score:.4f}")


def main() -> None:
    """
    @brief Main function to perform image classification using MobileNetV2.

    This script parses command-line arguments, prepares input image,
    performs inference using the MobileNetV2 model with HB_HBMRuntime backend,
    and prints top-K classification results.

    @return None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='/opt/hobot/model/x5/basic/mobilenetv2_224x224_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--test-img', type=str,
                        default='data/seperated_conv.png',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str,
                        default='data/ImageNet_1k.json',
                        help='Path to load ImageNet label file.')

    opt = parser.parse_args()

    # Automatically download model if not found
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading MobileNetV2 model...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                  "rdk_s100/MobileNet/mobilenetv2_224x224_nv12.hbm")
        opt.model_path = 'mobilenetv2_224x224_nv12.hbm'

    # Load class index-to-label mapping if available
    idx2label: Optional[Dict[int, str]] = None
    if os.path.exists(opt.label_file):
        with open(opt.label_file, "r") as f:
            idx2label = eval(f.read())  # Expected format: {index: label}

    # Instantiate MobileNetV2 model
    mobilenetv2 = MobileNetV2(opt)

    # Set inference priority and BPU core assignment
    mobilenetv2.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model info (e.g., input/output names and shapes)
    common.print_model_info(mobilenetv2.model)

    # Load and prepare image for inference
    img: np.ndarray = common.load_image(opt.test_img)
    input_array = mobilenetv2.pre_process(img)

    # Run inference
    outputs = mobilenetv2.forward(input_array)

    # Print classification result
    mobilenetv2.post_process(outputs, idx2label)


if __name__ == "__main__":
    main()
