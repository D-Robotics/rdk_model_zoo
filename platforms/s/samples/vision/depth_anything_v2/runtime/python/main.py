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

"""Depth Anything V2 monocular depth estimation sample.

This module provides a minimal CLI demo that loads a Depth Anything V2 HBM
model, runs NCHW RGB preprocessing, performs inference, and saves the
colorized depth map.

Typical Usage:
    >>> python main.py
    >>> python main.py --test-img ../../test_data/furseal.jpg
"""

import cv2
import os
import sys
import argparse

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
from depth_anything_v2 import DepthAnythingV2Config, DepthAnythingV2


def main() -> None:
    """Run Depth Anything V2 depth estimation demo."""
    default_model_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../model/s100/depth_any.hbm",
    ))
    default_test_img = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "../../test_data/furseal.jpg",
    ))

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=default_model_path,
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--test-img', type=str,
                        default=default_test_img,
                        help='Path to load the test image.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save the output depth map image.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes.")

    opt = parser.parse_args()

    # Init config
    config = DepthAnythingV2Config(
        model_path=opt.model_path
    )

    # Instantiate model
    model = DepthAnythingV2(config)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print model info
    inspect.print_model_info(model.model)

    # Load image
    img = file_io.load_image(opt.test_img)

    # Run inference
    depth_map = model.predict(img)

    # Colorize depth map
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    # Save result
    cv2.imwrite(opt.img_save_path, depth_color)
    print(f"[Saved] Depth map saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
