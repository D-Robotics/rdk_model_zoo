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

"""LaneNet lane detection inference entry script.

This script runs a BPU-quantized LaneNet (.hbm) model on a single input image
and saves instance segmentation and binary segmentation mask results.

Workflow:
    1) Parse CLI arguments.
    2) Check platform compatibility (S100 only).
    3) Download the model file if missing.
    4) Create LaneNetConfig and initialize LaneNet runtime wrapper.
    5) Preprocess image -> BPU inference -> postprocess masks.
    6) Save instance and binary segmentation results.

Notes:
    - This model only supports RDK S100 platform.
    - If running on RDK S600, inference will not produce correct results.
      Please refer to README.md for platform compatibility details.
    - The project root is appended to sys.path to import shared utilities
      under `utils/py_utils/`.

Example:
    python main.py \\
        --test-img ../../test_data/lane.jpg \\
        --instance-save-path instance_pred.png \\
        --binary-save-path binary_pred.png
"""

import os
import cv2
import sys
import argparse

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
from lanenet import LaneNet, LaneNetConfig


SUPPORTED_SOC = "s100"
MODEL_DOWNLOAD_URL = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/Lanenet/lanenet256x512.hbm"


def main() -> None:
    """Run LaneNet lane detection on a single image.

    This function parses command-line arguments, validates platform
    compatibility, loads the LaneNet model, preprocesses the input image,
    performs BPU inference, postprocesses segmentation results, and saves
    the output masks.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='/opt/hobot/model/s100/basic/lanenet256x512.hbm',
                        help='Path to BPU quantized *.hbm model file. Only S100 model is available.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/lane.jpg',
                        help='Path to load test image.')
    parser.add_argument('--instance-save-path', type=str, default='instance_pred.png',
                        help='Path to save instance segmentation result image.')
    parser.add_argument('--binary-save-path', type=str, default='binary_pred.png',
                        help='Path to save binary segmentation result image.')

    opt = parser.parse_args()

    # Platform compatibility check: LaneNet only supports S100
    if soc != SUPPORTED_SOC:
        print(f"[WARNING] Current platform: {soc}. LaneNet only supports RDK S100 (s100).")
        print(f"[WARNING] The model was trained and compiled for S100 BPU.")
        print(f"[WARNING] Inference results on {soc} may be incorrect or fail.")
        print(f"[WARNING] Please refer to README.md for platform compatibility details.")

    # Download model if missing (S100 only)
    file_io.download_model_if_needed(opt.model_path, MODEL_DOWNLOAD_URL)

    # Initialize LaneNet configuration and model
    config = LaneNetConfig(model_path=opt.model_path)
    lanenet = LaneNet(config)

    # Configure runtime scheduling (BPU cores, priority)
    lanenet.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(lanenet.model)

    # Load input image
    img = file_io.load_image(opt.test_img)

    # Preprocess image to match model input
    input_tensor = lanenet.pre_process(img)

    # Run inference
    outputs = lanenet.forward(input_tensor)

    # Postprocess outputs to get segmentation masks
    instance_pred, binary_pred = lanenet.post_process(outputs)

    # Save result images
    cv2.imwrite(opt.instance_save_path, instance_pred)
    cv2.imwrite(opt.binary_save_path, binary_pred)
    print(f"[Saved] Instance segmentation result: {opt.instance_save_path}")
    print(f"[Saved] Binary segmentation result:   {opt.binary_save_path}")


if __name__ == "__main__":
    main()
