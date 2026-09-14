# Copyright (c) 2026 D-Robotics Corporation
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

"""
MODNet Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for MODNet
portrait matting on a single input image, following the RDK Model Zoo
engineering standards.

Workflow:
    1) Parse CLI arguments for model paths, data, and parameters.
    2) Initialize MODNetConfig and MODNet model wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load image and execute full pipeline: Preprocess -> Forward -> Postprocess.
    5) Save the alpha matte and optionally composite with a background.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
from modnet import MODNet, MODNetConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s].%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MODNet")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "modnet_512x512_rgb.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "person.jpg")
DEFAULT_BG_IMAGE = os.path.join(TEST_DATA_DIR, "bg.jpg")
DEFAULT_MATTE_PATH = os.path.join(TEST_DATA_DIR, "matte.png")
DEFAULT_RESULT_PATH = os.path.join(TEST_DATA_DIR, "result.png")


def composite(image: np.ndarray, matte: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Composite the foreground (image) onto a background using the alpha matte.

    Args:
        image: Original image in BGR format.
        matte: Alpha matte as a uint8 grayscale image.
        background: Background image in BGR format.

    Returns:
        Composited image.
    """
    bg = cv2.resize(background, (image.shape[1], image.shape[0]))
    alpha = matte.astype(np.float32)[:, :, None] / 255.0
    result = (image.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha))
    return result.astype(np.uint8)


def save_image(path: str, image: np.ndarray) -> None:
    """Save the result image to disk."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """Run the complete MODNet matting pipeline on a single image."""
    parser = argparse.ArgumentParser(description="MODNet Portrait Matting Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--bg-img", type=str, default=DEFAULT_BG_IMAGE,
                        help="Path to the background image for compositing.")
    parser.add_argument("--matte-save-path", type=str, default=DEFAULT_MATTE_PATH,
                        help="Path to save the alpha matte.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_PATH,
                        help="Path to save the composited result image.")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0],
                        help="BPU core indexes to run inference.")
    parser.add_argument("--ref-size", type=int, default=512,
                        help="Target input resolution (longer side).")
    args = parser.parse_args()

    config = MODNetConfig(
        model_path=args.model_path,
        ref_size=args.ref_size,
    )
    model = MODNet(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    image = file_io.load_image(args.test_img)
    matte = model.predict(image)

    save_image(args.matte_save_path, matte)
    logger.info(f'Saved alpha matte to "{args.matte_save_path}"')

    if os.path.exists(args.bg_img):
        background = file_io.load_image(args.bg_img)
        result = composite(image, matte, background)
        save_image(args.img_save_path, result)
        logger.info(f'Saved composited result to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
