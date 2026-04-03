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
YOLO26 Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for YOLO26
on a single input image, following the RDK Model Zoo engineering standards.

Workflow:
    1) Parse CLI arguments for model, task, and runtime parameters.
    2) Initialize the corresponding YOLO26Config and task wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load image and labels, then execute the full pipeline.
    5) Visualize and save results for vision tasks, or print logs for classification.

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
from yolo26_det import YOLO26Detect, YOLO26Config
from yolo26_seg import YOLO26Seg, YOLO26SegConfig
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "datasets/coco/assets"))
RESULT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "yolo26n_detect_bayese_640x640_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "bus.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(RESULT_DIR, "result_detect.jpg")
DEFAULT_LABEL_FILES = {
    "detect": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "seg": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "pose": os.path.join(PROJECT_ROOT, "datasets/coco/coco_classes.names"),
    "cls": os.path.join(PROJECT_ROOT, "datasets/imagenet/imagenet_classes.names"),
    "obb": os.path.join(PROJECT_ROOT, "datasets/dotav1/dota_classes.names"),
}

logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to the target path.

    Args:
        path (str): Output image path.
        image (np.ndarray): Image array to save.

    Raises:
        RuntimeError: If OpenCV fails to write the image.
    """
    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    if not cv2.imwrite(path, image):
        raise RuntimeError(f"Failed to save image to {path}")


def main() -> None:
    """Run YOLO26 inference on a single image.

    This function orchestrates the complete inference process:
    - Argument parsing
    - Task wrapper initialization
    - Label and image loading
    - Runtime scheduling configuration
    - Inference execution
    - Result visualization and saving
    """
    parser = argparse.ArgumentParser(description="YOLO26 Inference")
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "seg", "pose", "cls", "obb"],
                        help="Task type to run.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the model file used by hbm_runtime.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--label-file", type=str, default="",
                        help="Path to the label file. Empty means using the default labels for the task.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE,
                        help="Path to save the output result image.")
    parser.add_argument("--priority", type=int, default=0,
                        help="Model priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0],
                        help="BPU core indexes used for inference.")
    parser.add_argument("--classes-num", type=int, default=80,
                        help="Number of classes used by detection-style tasks.")
    parser.add_argument("--score-thres", type=float, default=0.25,
                        help="Score threshold used to filter predictions.")
    parser.add_argument("--nms-thres", type=float, default=0.70,
                        help="NMS threshold used to suppress overlapping boxes.")
    parser.add_argument("--strides", type=str, default="8,16,32",
                        help="Comma-separated strides used by the decoder.")
    parser.add_argument("--topk", type=int, default=5,
                        help="Top-K results returned by classification.")
    parser.add_argument("--kpt-conf-thres", type=float, default=0.50,
                        help="Keypoint confidence threshold used by pose.")
    parser.add_argument("--angle-sign", type=float, default=1.0,
                        help="Angle sign used by OBB decoding.")
    parser.add_argument("--angle-offset", type=float, default=0.0,
                        help="Angle offset used by OBB decoding.")
    parser.add_argument("--regularize", type=int, default=1,
                        help="Whether to regularize OBB angles, 1 for enabled and 0 for disabled.")
    args = parser.parse_args()

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
        cfg = YOLO26Config(
            model_path=args.model_path,
            classes_num=args.classes_num,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            strides=strides,
        )
        model = YOLO26Detect(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        results = model.predict(img)
        visualize.draw_detect_yolo26(img, results, labels, visualize.rdk_colors)
        save_image(args.img_save_path, img)
        return

    if args.task == "seg":
        cfg = YOLO26SegConfig(
            model_path=args.model_path,
            classes_num=args.classes_num,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            strides=np.array(strides, dtype=np.int32),
        )
        model = YOLO26Seg(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        xyxy, score, cls, masks = model.predict(img)
        results = [{"box": box, "score": sc, "id": cid, "mask": mask.astype(np.uint8)}
                   for box, sc, cid, mask in zip(xyxy, score, cls, masks)]
        visualize.draw_seg_yolo26(img, results, labels, visualize.rdk_colors)
        save_image(args.img_save_path, img)
        return

    if args.task == "pose":
        cfg = YOLO26PoseConfig(
            model_path=args.model_path,
            score_thres=args.score_thres,
            nms_thres=args.nms_thres,
            kpt_conf_thres=args.kpt_conf_thres,
            strides=strides,
        )
        model = YOLO26Pose(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        results = model.predict(img)
        visualize.draw_pose_yolo26(img, results, visualize.COCO_SKELETON, args.kpt_conf_thres)
        save_image(args.img_save_path, img)
        return

    if args.task == "cls":
        cfg = YOLO26ClsConfig(
            model_path=args.model_path,
            topk=args.topk,
        )
        model = YOLO26Cls(cfg)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        inspect.print_model_info(model.model)
        results = model.predict(img)
        visualize.draw_cls_yolo26(img, results, labels)
        return

    cfg = YOLO26OBBConfig(
        model_path=args.model_path,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        angle_sign=args.angle_sign,
        angle_offset=args.angle_offset,
        regularize=bool(args.regularize),
        strides=strides,
    )
    model = YOLO26OBB(cfg)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    inspect.print_model_info(model.model)
    results = model.predict(img)
    visualize.draw_obb_yolo26(img, results, labels, visualize.rdk_colors)
    save_image(args.img_save_path, img)


if __name__ == "__main__":
    main()
