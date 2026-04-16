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
from typing import Dict, Optional

# Append parent directory to sys.path to enable local 'utils' module imports
sys.path.append(os.path.abspath("../../.."))
import utils.py_utils.preprocess_utils as pre_utils
import utils.py_utils.postprocess_utils as post_utils
import utils.py_utils.common_utils as common


class Resnet18:
    """
    @brief Wrapper class for running inference using a ResNet18 model through HB_HBMRuntime.
    """

    def __init__(self, opt):
        """
        @brief Initialize the Resnet18 model with model path and extract I/O details.

        @param opt (argparse.Namespace) Command-line or config object containing model_path.
        """
        # Load model runtime
        self.model = hbm_runtime.HB_HBMRuntime(opt.model_path)

        # Retrieve model name and input/output names
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.shapes = self.model.input_shapes[self.model_name]

        # Extract input resolution (Height, Width)
        # self.input_H = self.shapes[self.input_names[0]][1]
        # self.input_W = self.shapes[self.input_names[0]][2]
        self.input_H = self.shapes[self.input_names[0]][2]
        self.input_W = self.shapes[self.input_names[0]][3]

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """
        @brief Set optional scheduling parameters such as priority and BPU core assignment.

        @param priority (int, optional) Scheduling priority (0-255).
        @param bpu_cores (list[int], optional) List of BPU core indices to use for inference.
        @return None
        """
        kwargs = {}

        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}  # Set inference priority
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}  # Assign BPU cores

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
        @brief Run forward inference using the preprocessed input tensor.

        @param input_tensor (Dict[str, Dict[str, np.ndarray]]) Input data keyed by model and input name.
        @return Dict[str, np.ndarray]: Output tensors keyed by output name.
        """
        outputs = self.model.run(input_tensor)
        return outputs[self.model_name]

    def post_process(self,
                     outputs: Dict[str, np.ndarray],
                     idx2label: Dict[int, str]) -> None:
        """
        @brief Postprocess output and print top-K predicted labels.

        @param outputs (Dict[str, np.ndarray]) Output tensor dictionary from model inference.
        @param idx2label (Dict[int, str]) Mapping from class index to human-readable label.
        @return None
        """
        # Display top-K predictions from output
        post_utils.print_topk_predictions(outputs[self.output_names[0]][0], idx2label)


def main() -> None:
    """
    @brief Main function to perform image classification using ResNet18.

    This function loads model and label files, preprocesses an input image,
    performs inference using the HB_HBMRuntime backend, and prints classification results.

    @return None
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='/opt/hobot/model/x5/basic/resnet18_224x224_nv12.bin',
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""")
    parser.add_argument('--priority', type=int, default=0,
                        help="Model priority (0~255). 0 is lowest, 255 is highest.")
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--test-img', type=str,
                        default='data/white_wolf.JPEG',
                        help='Path to load the test image.')
    parser.add_argument('--label-file', type=str,
                        default='data/ImageNet_1k.json',
                        help='Path to load ImageNet label mapping file.')

    opt = parser.parse_args()




    # Download model file if not present
    if not os.path.exists(opt.model_path):
        print(f"File {opt.model_path} does not exist. Downloading ResNet18 model...")
        os.system("wget -c https://archive.d-robotics.cc/downloads/"
                  "rdk_model_zoo/rdk_x5/resnet18_224x224_nv12.bin")
        opt.model_path = 'resnet18_224x224_nv12.bin'

    # Load label mapping if available
    idx2label: Optional[Dict[int, str]] = None
    if os.path.exists(opt.label_file):
        with open(opt.label_file, "r") as f:
            idx2label = eval(f.read())  # Expected format: {index: label}

    # Initialize ResNet18 model instance
    resnet18 = Resnet18(opt)

    # Set runtime scheduling parameters
    resnet18.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model information (e.g., input/output names, shape)
    common.print_model_info(resnet18.model)

    # Load input image from disk
    img: np.ndarray = common.load_image(opt.test_img)

    # Preprocess image into NV12 format
    input_array = resnet18.pre_process(img)

    # Run inference on the preprocessed input
    outputs = resnet18.forward(input_array)

    # Print top-K classification results
    resnet18.post_process(outputs, idx2label)


if __name__ == "__main__":
    main()
