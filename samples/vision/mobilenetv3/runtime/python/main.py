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
MobileNetV3 image classification inference entry script.

This module provides the standard Python entry for the MobileNetV3 sample on
`RDK X5`. The script is responsible for parsing command-line arguments,
constructing the runtime wrapper, loading the input image and labels,
running inference, printing Top-K results, and saving the final
visualization image.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from mobilenetv3 import MobileNetV3, MobileNetV3Config


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MobileNetV3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetV3_224x224_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "kit_fox.JPEG")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result.jpg")
DEFAULT_LABEL_FILE = os.path.join(TEST_DATA_DIR, "ImageNet_1k.json")


def save_image(path: str, image) -> None:
    """Save the classification visualization image to disk."""

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """
    Run the complete MobileNetV3 classification pipeline on a single image.

    The entry follows the standardized sample pattern used in this repository:
    1. Parse default-usable command-line arguments.
    2. Build the MobileNetV3 runtime configuration.
    3. Load the ImageNet labels and the test image.
    4. Execute `predict()` on the runtime wrapper.
    5. Print Top-K results and save the visualization image.
    """

    parser = argparse.ArgumentParser(description="MobileNetV3 Classification Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--label-file", type=str, default=DEFAULT_LABEL_FILE, help="Path to the ImageNet label file.")
    parser.add_argument("--priority", type=int, default=0, help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes to run inference.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test input image.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save output result image.")
    parser.add_argument("--resize-type", type=int, default=0, help="Resize strategy (0: direct, 1: letterbox).")
    parser.add_argument("--topk", type=int, default=5, help="Number of top results to return.")
    args = parser.parse_args()

    config = MobileNetV3Config(
        model_path=args.model_path,
        label_file=args.label_file,
        resize_type=args.resize_type,
        topk=args.topk,
    )
    model = MobileNetV3(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    image = file_io.load_image(args.test_img)
    labels = model.labels
    topk_idx, topk_prob, topk_labels = model.predict(image)

    logger.info(f"Top-{args.topk} results:")
    for i, (cid, score, label) in enumerate(zip(topk_idx, topk_prob, topk_labels), start=1):
        logger.info(f"Rank {i}: class={cid}, label={label}, score={score:.4f}")

    vis_results = list(zip(topk_idx.tolist(), topk_prob.tolist()))
    vis_image = visualize.draw_classification(image.copy(), vis_results, labels)
    save_image(args.img_save_path, vis_image)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
