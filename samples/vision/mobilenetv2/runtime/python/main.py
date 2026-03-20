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

"""MobileNetV2 image classification sample.

This module provides a minimal CLI demo that loads a MobileNetV2 HBM model, runs
NV12 preprocessing, performs inference, and prints top-K ImageNet predictions.

Key Features:
    - CLI arguments for model path, priority, BPU cores, input image, and labels
    - Top-K results printed with label mapping from a text file

Typical Usage:
    >>> # Run with default paths (model may be downloaded automatically)
    >>> # python main.py
    >>>
    >>> # Specify image and show top-K results
    >>> # python main.py --test-img /path/to/image.jpg --label-file labels.txt

Notes:
    - The label file is expected to be a plain text file with one label per line,
      where the line index corresponds to the `class_id`.
"""

import os
import cv2
import sys
import argparse
import numpy as np
from typing import Optional, Dict

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess.py
#   utils/py_utils/visualize.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from mobilenetv2 import MobileNetV2Config, MobileNetV2


def main() -> None:
    """Run a MobileNetV2 image classification demo.

    This function parses command-line arguments, loads ImageNet labels from a text file,
    preprocesses an input image, runs inference via the HB_HBMRuntime backend,
    and prints top-K classification results.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                    default=f"/opt/hobot/model/{soc}/basic/mobilenetv2_224x224_nv12.hbm",
                    help="""Path to BPU *.hbm Model.""")
    parser.add_argument('--priority', type=int, default=0,
                        help="Model priority (0~255). 0 is lowest, 255 is highest.")
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run. Provide a list of integers (e.g., --bpu_cores 0 1). ")
    parser.add_argument('--test-img', type=str,
                        default='../../test_data/zebra_cls.jpg',
                        help='Path to load the test image.')
    parser.add_argument('--label-file', type=str,
                        default='../../test_data/imagenet1000_labels.txt',
                        help='Path to load ImageNet label mapping file.')

    opt = parser.parse_args()

    # Init config parameter
    config = MobileNetV2Config(
        model_path=opt.model_path
    )

    # Load label mapping if available
    idx2label: Optional[Dict[int, str]] = None
    if os.path.exists(opt.label_file):
        with open(opt.label_file, "r") as f:
            idx2label = {i: line.strip() for i, line in enumerate(f)}  # Expected format: {index: label}

    # Initialize MobileNetV2 model instance
    mobilenetv2 = MobileNetV2(config)

    # Set runtime scheduling parameters
    mobilenetv2.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model information (e.g., input/output names, shape)
    inspect.print_model_info(mobilenetv2.model)

    # Load input image from disk
    img: np.ndarray = file_io.load_image(opt.test_img)

    # Preprocess image into NV12 format
    input_array = mobilenetv2.pre_process(img)

    # Run inference on the preprocessed input
    outputs = mobilenetv2.forward(input_array)

    # Post-process model outputs to get top-K classification results
    cls_results = mobilenetv2.post_process(outputs, 5)

    # Print classification results using label names
    visualize.print_classification_results(cls_results, idx2label)


if __name__ == "__main__":
    main()
