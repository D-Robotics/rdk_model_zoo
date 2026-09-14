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

"""UnetMobileNet semantic segmentation image inference entry script.

This script runs a BPU-quantized UnetMobileNet (.hbm) model on a single input
image and produces a colorized segmentation overlay.

Workflow:
    1) Parse CLI arguments.
    2) Download the model file if missing (based on SoC type).
    3) Create UnetMobileNetConfig and initialize UnetMobileNet runtime wrapper.
    4) Preprocess image -> BPU inference -> postprocess segmentation.
    5) Save the blended result image.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory so that
      relative paths (e.g., `../../../../../`) resolve correctly.

Example:
    python main.py \\
        --test-img ../../test_data/segmentation.png \\
        --img-save-path result.jpg \\
        --alpha-f 0.75
"""

import os
import cv2
import sys
import argparse

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
from unetmobilenet import UnetMobileNet, UnetMobileNetConfig


def main() -> None:
    """Run UnetMobileNet semantic segmentation on a single image.

    This function parses command-line arguments, loads the UnetMobileNet model,
    preprocesses the input image, performs inference on the BPU,
    postprocesses the segmentation result, and saves the blended output image.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/unet_mobilenet_1024x2048_nv12.hbm',
                        help='Path to BPU Quantized *.hbm Model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/segmentation.png',
                        help='Path to load test image.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with segmentation overlay.')
    parser.add_argument('--alpha-f', type=float, default=0.75,
                        help='Alpha blending factor. 0.0 = only mask, 1.0 = only original image.')

    opt = parser.parse_args()

    # Select model download URL based on SoC platform
    if soc == "s600":
        download_url = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm"
    else:
        download_url = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/unetmobilenet/unet_mobilenet_1024x2048_nv12.hbm"

    file_io.download_model_if_needed(opt.model_path, download_url)

    # Init config
    config = UnetMobileNetConfig(
        model_path=opt.model_path,
        alpha_f=opt.alpha_f
    )

    # Instantiate UnetMobileNet model
    unet = UnetMobileNet(config)

    # Configure runtime scheduling (BPU cores, priority)
    unet.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(unet.model)

    # Load input image
    img = file_io.load_image(opt.test_img)

    # Run full pipeline: preprocess -> inference -> postprocess -> visualize
    blended_img = unet.predict(img, alpha_f=opt.alpha_f)

    # Save the resulting image
    cv2.imwrite(opt.img_save_path, blended_img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
