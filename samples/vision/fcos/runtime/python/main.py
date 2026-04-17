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
FCOS detection inference entry script.

This module provides the standard Python entry for the FCOS sample on RDK X5.
The script is responsible for parsing command-line arguments, loading labels
and the input image, constructing the FCOS runtime wrapper, executing
inference, and saving the final visualization result.
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
from fcos_det import FCOSConfig, FCOSDetect


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FCOS")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "fcos_efficientnetb0_detect_512x512_bayese_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "bus.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result.jpg")
DEFAULT_LABEL_FILE = os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names")


def save_image(path: str, image) -> None:
    """
    Save the visualization result to the target path.

    Args:
        path (str): Absolute or relative path of the output image.
        image (np.ndarray): Visualization image to save.

    Raises:
        RuntimeError: Raised when OpenCV fails to write the image.
    """

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """
    Run the complete FCOS detection pipeline on a single image.

    The entry follows the standardized sample pattern used in this repository:
    1. Parse default-usable command-line arguments.
    2. Build the FCOS runtime configuration.
    3. Load labels and the test image.
    4. Execute `predict()` on the runtime wrapper.
    5. Draw detections and save the output image.
    """

    parser = argparse.ArgumentParser(description="FCOS Detection Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--label-file", type=str, default=DEFAULT_LABEL_FILE, help="Path to the class names file.")
    parser.add_argument("--priority", type=int, default=0, help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes to run inference.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test input image.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save output result image.")
    parser.add_argument("--resize-type", type=int, default=0, help="Resize strategy (0: direct, 1: letterbox).")
    parser.add_argument("--classes-num", type=int, default=80, help="Number of detection classes.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold used to filter predictions.")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU threshold used to suppress overlapping boxes.")
    parser.add_argument("--strides", type=lambda s: list(map(int, s.split(","))), default=[8, 16, 32, 64, 128], help="Comma-separated strides used by FCOS heads.")
    args = parser.parse_args()

    config = FCOSConfig(
        model_path=args.model_path,
        classes_num=args.classes_num,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        resize_type=args.resize_type,
        strides=args.strides,
    )
    model = FCOSDetect(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    labels = file_io.load_class_names(args.label_file) if os.path.exists(args.label_file) else []
    image = file_io.load_image(args.test_img)
    boxes, scores, cls_ids = model.predict(image)

    visualize.draw_detection_results(image, boxes, cls_ids, scores, labels)
    save_image(args.img_save_path, image)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
