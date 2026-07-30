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
PaddleOCR Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for PaddleOCR
on a single input image, following the RDK Model Zoo engineering standards.

The pipeline uses two models:
  - Detection model: DB text detection with NV12 input.
  - Recognition model: CTC text recognition with float32 featuremap input.

Workflow:
    1) Parse CLI arguments for model paths, data, and parameters.
    2) Initialize PaddleOCRConfig and PaddleOCR model wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load image and execute full pipeline: Detection -> Crop -> Recognition.
    5) Visualize and save the resulting image with detected boxes and text.
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
from paddleocr import PaddleOCR, PaddleOCRConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s].%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PaddleOCR")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_DET_MODEL_PATH = os.path.join(MODEL_DIR, "en_PP-OCRv3_det_640x640_nv12.bin")
DEFAULT_REC_MODEL_PATH = os.path.join(MODEL_DIR, "en_PP-OCRv3_rec_48x320_rgb.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "paddleocr_test.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result.jpg")


def draw_boxes_and_texts(
    image: np.ndarray,
    boxes: list,
    texts: list,
) -> np.ndarray:
    """Draw detected boxes on the image and recognized texts on a white panel.

    Args:
        image (np.ndarray): Original image.
        boxes (list): List of 4-point bounding boxes.
        texts (list): List of recognized text strings.

    Returns:
        np.ndarray: Combined image with boxes and text panel.
    """
    img_boxes = image.copy()
    for box in boxes:
        box_int = box.astype(int)
        cv2.polylines(img_boxes, [box_int], isClosed=True, color=(128, 240, 128), thickness=3)

    white_panel = np.ones(image.shape, dtype=np.uint8) * 255
    y_offset = 60
    for i, text in enumerate(texts):
        cv2.putText(
            white_panel,
            text,
            (10, y_offset + i * 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            3,
        )

    return np.hstack((img_boxes, white_panel))


def save_image(path: str, image) -> None:
    """Save the result image to disk."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """Run the complete PaddleOCR pipeline on a single image.

    The entry follows the standardized sample pattern used in this repository:
    1. Parse default-usable command-line arguments.
    2. Build the PaddleOCR runtime configuration.
    3. Load the test image.
    4. Execute `predict()` on the runtime wrapper.
    5. Print recognized texts and save the visualization image.
    """
    parser = argparse.ArgumentParser(description="PaddleOCR Inference")
    parser.add_argument("--det-model-path", type=str, default=DEFAULT_DET_MODEL_PATH,
                        help="Path to the detection model (.bin).")
    parser.add_argument("--rec-model-path", type=str, default=DEFAULT_REC_MODEL_PATH,
                        help="Path to the recognition model (.bin).")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--det-threshold", type=float, default=0.5,
                        help="Binarization threshold for detection output.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE,
                        help="Path to save output result image.")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0],
                        help="BPU core indexes to run inference.")
    args = parser.parse_args()

    config = PaddleOCRConfig(
        det_model_path=args.det_model_path,
        rec_model_path=args.rec_model_path,
        det_threshold=args.det_threshold,
    )
    model = PaddleOCR(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.det_model)
    inspect.print_model_info(model.rec_model)

    image = file_io.load_image(args.test_img)
    boxes, texts = model.predict(image)

    logger.info(f"Detected {len(boxes)} text regions:")
    for i, (box, text) in enumerate(zip(boxes, texts), start=1):
        logger.info(f"  [{i}] {text}")

    vis_image = draw_boxes_and_texts(image, boxes, texts)
    save_image(args.img_save_path, vis_image)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
