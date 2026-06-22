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

"""EfficientNet-Lite image classification command line sample.

The entry script parses arguments, loads the test image and labels, builds a
configuration object, runs ``predict()``, and prints Top-K ImageNet results.
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
from efficientnet import EfficientNet, EfficientNetConfig


def main() -> None:
    """Run an EfficientNet-Lite image classification demo."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='../../model/s100/efficientnet_lite0_224x224_nv12.hbm',
                        help='Path to the compiled HBM model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str,
                        default='../../test_data/Scottish_deerhound.JPEG',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/imagenet_classes.names',
                        help='Path to load label file.')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top predictions to display.')
    parser.add_argument('--resize-type', type=int, default=1,
                        help='Resize strategy: 0 (direct), 1 (letterbox).')

    opt = parser.parse_args()

    config = EfficientNetConfig(
        model_path=opt.model_path,
        resize_type=opt.resize_type
    )

    efficientnet = EfficientNet(config)
    efficientnet.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    inspect.print_model_info(efficientnet.model)

    idx2label: Dict[int, str] = {}
    if os.path.exists(opt.label_file):
        idx2label = file_io.load_labels(opt.label_file)

    img: np.ndarray = file_io.load_image(opt.test_img)
    cls_results = efficientnet.predict(img, topk=opt.top_k)

    visualize.print_classification_results(cls_results, idx2label)


if __name__ == "__main__":
    main()
