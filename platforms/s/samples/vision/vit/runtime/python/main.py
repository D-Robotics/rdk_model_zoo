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

"""ViT CIFAR-10 image classification command line sample.

The entry script parses arguments, loads the test image and labels, builds a
configuration object, runs ``predict()``, and prints Top-K CIFAR-10 results.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from vit import ViTConfig, ViT


def main() -> None:
    """Run a ViT image classification demo."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-variant', type=str, default='int8',
                        choices=['int8', 'int16'],
                        help='Model variant: int8 or int16 quantization.')
    parser.add_argument('--model-path', type=str,
                        default='../../model/s100/vit_cifar10_batch1_int8.hbm',
                        help='Path to the compiled HBM model.')
    parser.add_argument('--priority', type=int, default=0,
                        help="Model priority (0~255). 0 is lowest, 255 is highest.")
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu-cores 0 1).")
    parser.add_argument('--test-img', type=str,
                        default='../../test_data/airplane_0000.png',
                        help='Path to load the test image.')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/cifar10_classes.names',
                        help='Path to load CIFAR-10 label mapping file.')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of classification results to print.')

    opt = parser.parse_args()

    if opt.model_variant == 'int16' and opt.model_path == parser.get_default('model_path'):
        opt.model_path = '../../model/s100/vit_cifar10_batch1_int16.hbm'

    config = ViTConfig(
        model_path=opt.model_path
    )

    idx2label: Dict[int, str] = {}
    if os.path.exists(opt.label_file):
        idx2label = file_io.load_labels(opt.label_file)

    vit = ViT(config)

    vit.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    inspect.print_model_info(vit.model)

    img: np.ndarray = file_io.load_image(opt.test_img)

    cls_results = vit.predict(img, topk=opt.top_k)

    visualize.print_classification_results(cls_results, idx2label)


if __name__ == "__main__":
    main()
