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
YOLOWorld open-vocabulary detection entry script.

This module provides the standard Python entry for the YOLOWorld sample on
RDK X5. It loads offline vocabulary embeddings, runs open-vocabulary detection
for selected prompts, draws the result, and saves the visualization image.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

import cv2
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yoloworld_det import YOLOWorldConfig, YOLOWorldDetect


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YOLOWorld")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yolo_world.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "dog.jpeg")
DEFAULT_VOCAB_FILE = os.path.join(TEST_DATA_DIR, "offline_vocabulary_embeddings.json")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "inference.png")
DEFAULT_PROMPTS = "dog"


def parse_prompts(prompts: str) -> List[str]:
    """
    Parse comma-separated prompt words.

    Args:
        prompts: Comma-separated prompt string.

    Returns:
        Prompt list with empty items removed.
    """

    return [item.strip() for item in prompts.split(",") if item.strip()]


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save an output visualization image.

    Args:
        path: Target image path.
        image: Image array to save.

    Raises:
        RuntimeError: If OpenCV fails to write the image.
    """

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def draw_results(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    cls_ids: np.ndarray,
    class_names: List[str],
) -> np.ndarray:
    """
    Draw YOLOWorld detection results on the original image.

    Args:
        image: Original BGR image.
        boxes: Detection boxes in original image coordinates.
        scores: Detection confidence scores.
        cls_ids: Vocabulary class IDs.
        class_names: Full vocabulary name list.

    Returns:
        Visualization image with detection results.
    """

    result = image.copy()
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        x1, y1, x2, y2 = box.astype(int)
        name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
        logger.info(f"({x1}, {y1}, {x2}, {y2}) -> {name}: {score:.2f}")
        color = visualize.rdk_colors[int(cls_id) % len(visualize.rdk_colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(result, f"{name}: {score:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return result


def main() -> None:
    """
    Run YOLOWorld open-vocabulary detection on one image.

    The entry follows the standard sample pattern: parse arguments, construct
    the wrapper, load image and prompts, run `predict()`, then save the result.
    """

    parser = argparse.ArgumentParser(description="YOLOWorld Open-Vocabulary Detection")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--vocab-file", type=str, default=DEFAULT_VOCAB_FILE, help="Path to offline vocabulary embeddings JSON.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test input image.")
    parser.add_argument("--prompts", type=str, default=DEFAULT_PROMPTS, help="Comma-separated prompt words used for detection.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save output result image.")
    parser.add_argument("--priority", type=int, default=0, help="Model priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes used for inference.")
    parser.add_argument("--score-thres", type=float, default=0.05, help="Score threshold used to filter predictions.")
    parser.add_argument("--nms-thres", type=float, default=0.45, help="NMS threshold used to suppress overlapping boxes.")
    args = parser.parse_args()

    prompts = parse_prompts(args.prompts)
    image = file_io.load_image(args.test_img)
    config = YOLOWorldConfig(
        model_path=args.model_path,
        vocab_file=args.vocab_file,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
    )
    model = YOLOWorldDetect(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    boxes, scores, cls_ids = model.predict(image, prompts)
    result_image = draw_results(image, boxes, scores, cls_ids, model.class_names)
    save_image(args.img_save_path, result_image)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
