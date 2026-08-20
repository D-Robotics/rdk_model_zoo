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

"""EfficientSAM-Tiny dual-bin full-mask inference entry script."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

from efficient_sam import EfficientSAMConfig, EfficientSAMSegment

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))
from utils.py_utils import file_io, inspect, visualize  # shared Model Zoo utils: image I/O, board detection, viz

logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("EfficientSAM")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_TEST_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "dogs.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "efficient_sam_full_mask_result.jpg")
# Output mask uses a distinct name from the committed reference
# ``test_data/efficient_sam_binary_mask.png`` so the reference is preserved for diffing.
DEFAULT_MASK_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "efficient_sam_binary_mask_result.png")


def save_outputs(result: dict, image: np.ndarray, result_path: str, mask_path: str) -> None:
    """Save the overlay image and binary mask from a prediction result.

    Args:
        result: Prediction dictionary returned by `EfficientSAMSegment.predict`.
        image: Original BGR input image.
        result_path: Destination path for the mask-overlay visualization.
        mask_path: Destination path for the binary mask image.

    Raises:
        RuntimeError: If either output image cannot be written.
    """

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    overlay = visualize.draw_mask_result(
        image, result["mask"], result["iou"], result["mask_index"],
        title="EfficientSAM full mask: encoder.hbm + decoder.hbm",
        color=(0, 180, 0),
    )
    file_io.save_image(result_path, overlay)
    file_io.save_image(mask_path, result["mask"].astype("uint8") * 255)


def main() -> None:
    """Parse CLI arguments and run the complete EfficientSAM-Tiny demo."""

    parser = argparse.ArgumentParser(description="EfficientSAM-Tiny dual-HBM full mask demo")
    parser.add_argument("--encoder-model-path", type=str, default=None, help="Override S-series encoder .hbm path.")
    parser.add_argument("--decoder-model-path", type=str, default=None, help="Override S-series decoder .hbm path.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to input image.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save mask overlay image.")
    parser.add_argument("--mask-save-path", type=str, default=DEFAULT_MASK_IMAGE, help="Path to save binary mask image.")
    parser.add_argument("--priority", type=int, default=0, help="Model scheduling priority.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes.")
    args = parser.parse_args()

    image = file_io.load_image(args.test_img)

    if args.encoder_model_path is None or args.decoder_model_path is None:
        board_type = Path("/sys/class/boardinfo/board_type")
        board_text = board_type.read_text(encoding="utf-8") if board_type.exists() else ""
        _, march, suffix, _ = inspect.resolve_platform(inspect.get_soc_name(), board_text)
        model_dir = os.path.join(SAMPLE_DIR, "model", march)
        args.encoder_model_path = args.encoder_model_path or os.path.join(
            model_dir, f"efficient_sam_vitt_encoder_512x512_{suffix}.hbm")
        args.decoder_model_path = args.decoder_model_path or os.path.join(
            model_dir, f"efficient_sam_vitt_decoder_512_{suffix}.hbm")
    config = EfficientSAMConfig(encoder_model_path=args.encoder_model_path, decoder_model_path=args.decoder_model_path)
    model = EfficientSAMSegment(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    result = model.predict(image)
    save_outputs(result, image, args.img_save_path, args.mask_save_path)
    logger.info('Saving results to "%s"', args.img_save_path)
    logger.info("Predicted IoU %.4f, mask index %d", result["iou"], result["mask_index"])


if __name__ == "__main__":
    main()
