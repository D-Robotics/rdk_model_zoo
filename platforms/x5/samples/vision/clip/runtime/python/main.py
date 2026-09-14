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
CLIP image-text matching entry script.

This module provides the standard Python entry for the CLIP sample on RDK X5.
It loads a test image and text prompts, runs the BPU image encoder plus ONNX
text encoder, prints similarity scores, and saves a visualization image.
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
from clip_retrieval import CLIPConfig, CLIPMatcher


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("CLIP")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "img_encoder.bin")
DEFAULT_TEXT_MODEL_PATH = os.path.join(MODEL_DIR, "text_encoder.onnx")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "dog.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "inference.png")
DEFAULT_TEXTS = "a diagram,a dog"


def parse_texts(texts: str) -> List[str]:
    """
    Parse comma-separated text prompts from the command line.

    Args:
        texts: Comma-separated prompt string.

    Returns:
        Prompt list with empty items removed.
    """

    return [item.strip() for item in texts.split(",") if item.strip()]


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


def draw_scores(image: np.ndarray, texts: List[str], scores: np.ndarray, order: np.ndarray) -> np.ndarray:
    """
    Draw CLIP similarity scores on the input image.

    Args:
        image: Original BGR image.
        texts: Candidate text prompts.
        scores: Similarity scores.
        order: Descending rank indexes.

    Returns:
        Visualization image with scores rendered at the top-left corner.
    """

    canvas = image.copy()
    for rank, idx in enumerate(order, start=1):
        text = f"Rank {rank}: {texts[int(idx)]} | similarity: {scores[int(idx)]:.4f}"
        logger.info(text)
        cv2.putText(canvas, text, (10, 40 + (rank - 1) * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return canvas


def main() -> None:
    """
    Run CLIP image-text matching on one image.

    The entry follows the standard sample pattern: parse arguments, construct
    the runtime wrapper, load image and text prompts, execute `predict()`, then
    save the visualization image.
    """

    parser = argparse.ArgumentParser(description="CLIP Image-Text Matching")
    parser.add_argument("--image-model-path", type=str, default=DEFAULT_IMAGE_MODEL_PATH, help="Path to the BPU image encoder *.bin model.")
    parser.add_argument("--text-model-path", type=str, default=DEFAULT_TEXT_MODEL_PATH, help="Path to the ONNX text encoder model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test input image.")
    parser.add_argument("--texts", type=str, default=DEFAULT_TEXTS, help="Comma-separated text prompts used for matching.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save output result image.")
    parser.add_argument("--priority", type=int, default=0, help="Image encoder priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes used by the image encoder.")
    args = parser.parse_args()

    texts = parse_texts(args.texts)
    image = file_io.load_image(args.test_img)
    config = CLIPConfig(image_model_path=args.image_model_path, text_model_path=args.text_model_path)
    model = CLIPMatcher(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.image_model)

    scores, order = model.predict(image, texts)
    result_image = draw_scores(image, texts, scores, order)
    save_image(args.img_save_path, result_image)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
