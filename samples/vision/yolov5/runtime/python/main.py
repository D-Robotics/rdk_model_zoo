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

"""
YOLOv5 Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for YOLOv5
on a single input image, following the RDK Model Zoo engineering standards.

Workflow:
    1) Parse CLI arguments for model, data, and parameters.
    2) Initialize YOLOv5Config and YOLOv5 model wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load data and execute full pipeline: Preprocess -> Forward -> Postprocess.
    5) Visualize and save the resulting image with detection boxes.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory.
    - RDK X5 uses `.bin` model files with `hbm_runtime`.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolov5_det import YOLOv5Config, YOLOv5Detect


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "bus.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result.jpg")
DEFAULT_LABEL_FILE = os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names")


def save_image(path: str, image: np.ndarray) -> None:
    """
    Save an image to the target path.

    Args:
        path (str): Output image path.
        image (np.ndarray): Image array to save.
    """

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """
    Run the complete YOLOv5 detection pipeline on a single image.

    This function parses command-line arguments, initializes the runtime
    wrapper, executes inference, visualizes the detections, and saves the
    final image to the requested output path.
    """

    parser = argparse.ArgumentParser(description="YOLOv5 Detection Inference")

    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the BPU quantized *.bin model.",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default=DEFAULT_LABEL_FILE,
        help="Path to the COCO class names file.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Model priority (0~255).",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        default=[0],
        help="BPU core indexes to run inference.",
    )
    parser.add_argument(
        "--test-img",
        type=str,
        default=DEFAULT_TEST_IMAGE,
        help="Path to the test input image.",
    )
    parser.add_argument(
        "--img-save-path",
        type=str,
        default=DEFAULT_RESULT_IMAGE,
        help="Path to save output result image.",
    )
    parser.add_argument(
        "--resize-type",
        type=int,
        default=0,
        help="Resize strategy (0: direct, 1: letterbox).",
    )
    parser.add_argument(
        "--classes-num",
        type=int,
        default=80,
        help="Number of detection classes.",
    )
    parser.add_argument(
        "--score-thres",
        type=float,
        default=0.25,
        help="Confidence threshold used to filter predictions.",
    )
    parser.add_argument(
        "--nms-thres",
        type=float,
        default=0.45,
        help="NMS threshold used to suppress overlapping boxes.",
    )
    parser.add_argument(
        "--anchors",
        type=lambda s: list(map(int, s.split(","))),
        default=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        help="Comma-separated anchor values used by YOLOv5.",
    )
    parser.add_argument(
        "--strides",
        type=lambda s: list(map(int, s.split(","))),
        default=[8, 16, 32],
        help="Comma-separated strides used by the detection heads.",
    )
    args = parser.parse_args()

    config = YOLOv5Config(
        model_path=args.model_path,
        classes_num=args.classes_num,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        resize_type=args.resize_type,
        anchors=args.anchors,
        strides=args.strides,
    )
    model = YOLOv5Detect(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    labels = file_io.load_class_names(args.label_file) if os.path.exists(args.label_file) else []
    image = file_io.load_image(args.test_img)
    results = model.predict(image)

    boxes = np.array([[x1, y1, x2, y2] for _, _, x1, y1, x2, y2 in results], dtype=np.float32)
    scores = np.array([score for _, score, *_ in results], dtype=np.float32)
    cls_ids = np.array([class_id for class_id, *_ in results], dtype=np.int32)

    visualize.draw_boxes(image, boxes, cls_ids, scores, labels, visualize.rdk_colors)
    save_image(args.img_save_path, image)
    print(f"[Info] Saving results to {args.img_save_path}")


if __name__ == "__main__":
    main()
