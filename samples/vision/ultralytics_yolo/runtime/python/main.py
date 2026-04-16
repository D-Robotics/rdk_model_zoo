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
Ultralytics YOLO Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for Ultralytics
YOLO on a single input image, following the RDK Model Zoo engineering
standards.

Workflow:
    1) Parse CLI arguments for model, task, and runtime parameters.
    2) Initialize the corresponding configuration and task wrapper.
    3) Configure runtime scheduling with BPU cores and priority.
    4) Load image and labels, then execute the full pipeline.
    5) Visualize and save results for vision tasks, or print Top-K logs for
       classification.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory.
    - RDK X5 uses `.bin` model files with `hbm_runtime`.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

# Add project root to sys.path so we can import shared utility modules.
sys.path.append(os.path.abspath("../../../../../"))

import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from ultralytics_yolo_cls import UltralyticsYOLOCls, UltralyticsYOLOClsConfig
from ultralytics_yolo_det import UltralyticsYOLODetect, UltralyticsYOLODetectConfig
from ultralytics_yolo_pose import UltralyticsYOLOPose, UltralyticsYOLOPoseConfig
from ultralytics_yolo_seg import UltralyticsYOLOSeg, UltralyticsYOLOSegConfig


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n_detect_bayese_640x640_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "bus.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(TEST_DATA_DIR, "result_detect.jpg")
DEFAULT_LABEL_FILES = {
    "detect": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "seg": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "pose": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "cls": os.path.join(PROJECT_ROOT, "datasets/imagenet/imagenet_classes.names"),
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Ultralytics_YOLO")


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to the target path.

    Args:
        path: Output image path.
        image: Image array to save.

    Raises:
        RuntimeError: If OpenCV fails to write the image.
    """
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """Run Ultralytics YOLO inference on a single image.

    This function orchestrates the complete inference process:
    - Argument parsing
    - Task wrapper initialization
    - Label and image loading
    - Runtime scheduling configuration
    - Inference execution
    - Result visualization and saving
    """
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Inference")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "seg", "pose", "cls"],
                        help="Task type to run.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the model file used by hbm_runtime.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--label-file", type=str, default="",
                        help="Path to the label file. Empty means using the default labels for the task.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE,
                        help="Path to save the output result image for detect, seg, and pose tasks.")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0],
                        help="BPU core indexes used for inference.")
    parser.add_argument("--classes-num", type=int, default=80,
                        help="Number of classes used by detection-style tasks.")
    parser.add_argument("--score-thres", type=float, default=0.25,
                        help="Score threshold used to filter predictions.")
    parser.add_argument("--nms-thres", type=float, default=0.70,
                        help="NMS threshold used to suppress overlapping boxes.")
    parser.add_argument("--strides", type=str, default="8,16,32",
                        help="Comma-separated strides used by the decoder.")
    parser.add_argument("--reg", type=int, default=16,
                        help="DFL regression channel count.")
    parser.add_argument("--mc", type=int, default=32,
                        help="Segmentation mask coefficient count.")
    parser.add_argument("--nkpt", type=int, default=17,
                        help="Number of pose keypoints.")
    parser.add_argument("--kpt-conf-thres", type=float, default=0.50,
                        help="Keypoint confidence threshold used by pose visualization.")
    parser.add_argument("--topk", type=int, default=5,
                        help="Top-K results returned by classification.")
    parser.add_argument("--resize-type", type=int, default=1,
                        help="Resize policy, 0 for direct resize and 1 for letterbox.")
    args = parser.parse_args()
    logger.info(args)

    label_file = args.label_file or DEFAULT_LABEL_FILES[args.task]
    strides = [int(item) for item in args.strides.split(",")]

    img = file_io.load_image(args.test_img)
    labels = []
    if os.path.exists(label_file):
        if args.task == "cls":
            labels = file_io.load_imagenet_labels(label_file)
        else:
            labels = file_io.load_class_names(label_file)

    if args.task == "detect":
        cfg = UltralyticsYOLODetectConfig(
            model_path=args.model_path,
            classes_num=args.classes_num,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            reg=args.reg,
            resize_type=args.resize_type,
            strides=strides,
        )
        model = UltralyticsYOLODetect(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        boxes, scores, cls_ids = model.predict(img)
        visualize.draw_detection_results(img, boxes, cls_ids, scores, labels, visualize.rdk_colors)
        save_image(args.img_save_path, img)
        return

    if args.task == "seg":
        cfg = UltralyticsYOLOSegConfig(
            model_path=args.model_path,
            classes_num=args.classes_num,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            reg=args.reg,
            mc=args.mc,
            resize_type=args.resize_type,
            strides=strides,
        )
        model = UltralyticsYOLOSeg(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        boxes, scores, cls_ids, masks = model.predict(img)
        visualize.draw_masks(img, boxes, masks, cls_ids, visualize.rdk_colors)
        visualize.draw_boxes(img, boxes, cls_ids, scores, labels, visualize.rdk_colors)
        save_image(args.img_save_path, img)
        return

    if args.task == "pose":
        cfg = UltralyticsYOLOPoseConfig(
            model_path=args.model_path,
            classes_num=1,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            reg=args.reg,
            nkpt=args.nkpt,
            resize_type=args.resize_type,
            strides=strides,
        )
        model = UltralyticsYOLOPose(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        boxes, scores, kpts = model.predict(img)
        visualize.draw_pose(
            img,
            boxes,
            kpts,
            skeleton=visualize.COCO_SKELETON,
            kpt_conf_thres=args.kpt_conf_thres,
            scores=scores,
        )
        save_image(args.img_save_path, img)
        return

    cfg = UltralyticsYOLOClsConfig(
        model_path=args.model_path,
        topk=args.topk,
        resize_type=args.resize_type,
    )
    model = UltralyticsYOLOCls(cfg)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    inspect.print_model_info(model.model)
    results = model.predict(img)
    logger.info("Top-%d results:", len(results))
    for rank, (class_id, score) in enumerate(results, start=1):
        class_name = labels.get(class_id, str(class_id))
        logger.info("Rank %d -> id: %d, score: %.4f, name: %s", rank, class_id, score, class_name)


if __name__ == "__main__":
    main()
