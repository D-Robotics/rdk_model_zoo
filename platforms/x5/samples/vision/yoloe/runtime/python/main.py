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
YOLOE Segmentation Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for YOLOE-11
instance segmentation (Prompt-Free) on a single input image, following the
RDK Model Zoo engineering standards.

Workflow:
    1) Parse CLI arguments for model paths, data, and parameters.
    2) Initialize YOLOESegConfig and YOLOESeg model wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load image and execute full pipeline: Preprocess -> Forward -> Postprocess.
    5) Visualize and save the resulting image with boxes and masks.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yoloe_seg import YOLOESeg, YOLOESegConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s].%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("YOLOE_Seg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yoloe_11s_seg_pf_bayese_640x640_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(PROJECT_ROOT, "datasets/coco/assets/bus.jpg")
DEFAULT_LABEL_FILE = os.path.join(PROJECT_ROOT, "datasets/yoloe/yoloe_seg_pf_classes.names")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result_seg.jpg")


def save_image(path: str, image: np.ndarray) -> None:
    """Save the result image to disk."""
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """Run the complete YOLOE segmentation pipeline on a single image."""
    parser = argparse.ArgumentParser(description="YOLOE Instance Segmentation Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--label-file", type=str, default=DEFAULT_LABEL_FILE,
                        help="Path to the class names file.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE,
                        help="Path to save output result image.")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0],
                        help="BPU core indexes to run inference.")
    parser.add_argument("--classes-num", type=int, default=4585,
                        help="Number of classes.")
    parser.add_argument("--score-thres", type=float, default=0.25,
                        help="Confidence threshold.")
    parser.add_argument("--nms-thres", type=float, default=0.70,
                        help="IoU threshold for NMS.")
    parser.add_argument("--strides", type=str, default="8,16,32",
                        help="Detection head strides, comma-separated.")
    parser.add_argument("--reg", type=int, default=16,
                        help="DFL bin count per side.")
    parser.add_argument("--mc", type=int, default=32,
                        help="Mask coefficient dimension.")
    args = parser.parse_args()

    strides = np.array([int(s) for s in args.strides.split(",")], dtype=np.int32)

    config = YOLOESegConfig(
        model_path=args.model_path,
        classes_num=args.classes_num,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        strides=strides,
        reg=args.reg,
        mc=args.mc,
    )
    model = YOLOESeg(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    img = file_io.load_image(args.test_img)
    labels = file_io.load_class_names(args.label_file)
    if len(labels) != args.classes_num:
        raise ValueError(f"Expected {args.classes_num} class names, got {len(labels)}")

    xyxy, score, cls, masks = model.predict(img)

    logger.info(f"Detected {len(xyxy)} instances:")
    results = []
    for i in range(len(xyxy)):
        cid = int(cls[i])
        cname = labels[cid] if cid < len(labels) else f"ID {cid}"
        logger.info(f"  [{i+1}] {cname}: {score[i]:.2f}")
        results.append({
            "box": xyxy[i].astype(int),
            "score": float(score[i]),
            "id": cid,
            "mask": masks[i].astype(np.uint8),
        })

    vis_img = img.copy()
    visualize.draw_seg_yolo26(vis_img, results, labels, visualize.rdk_colors)
    save_image(args.img_save_path, vis_img)
    logger.info(f'Saving results to "{args.img_save_path}"')


if __name__ == "__main__":
    main()
