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

"""Ultralytics YOLO Unified Inference Entry Script.

This script supports multiple Ultralytics YOLO task models. It dispatches to
the appropriate wrapper based on --task and uses --model-path to select the
actual model file.

Supported model variants:
    - yolov5u: YOLOv5u (anchor-free DFL head, detect)
    - yolov8:  YOLOv8
    - yolov10: YOLOv10 (NMS-free detection)
    - yolo11:  YOLO11
    - yolo12:  YOLO12

Supported tasks:
    - detect: Object detection
    - seg:    Instance segmentation
    - pose:   Pose estimation
    - cls:    Image classification

Example:
    python main.py --task detect --test-img ../../test_data/bus.jpg
    python main.py --task seg --model-path ../../model/nash-m/yolov8n_seg_nashm_640x640_nv12.hbm
    python main.py --task detect --model-path ../../model/nash-m/yolo12n_detect_nashm_640x640_nv12.hbm
    python main.py --task cls --model-path ../../model/nash-m/yolo11n_cls_nashm_640x640_nv12.hbm
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))

from yolo_detect import YoloDetect, YoloDetectConfig
from yolo_seg import YoloSeg, YoloSegConfig
from yolo_pose import YoloPose, YoloPoseConfig
from yolo_cls import YoloCls, YoloClsConfig
from yolo_v10detect import YoloV10Detect, YoloV10DetectConfig

import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as visualize
import utils.py_utils.inspect as inspect


# Model file naming conventions per variant and task.
# These names match the public S100 model archive; unsupported task/model
# combinations are intentionally omitted instead of guessed.
MODEL_FILE_PATTERNS = {
    "yolov5u": {
        "detect": "yolov5nu_detect_{suffix}_640x640_nv12.hbm",
    },
    "yolov8": {
        "detect": "yolov8n_detect_{suffix}_640x640_nv12.hbm",
        "seg":    "yolov8n_seg_{suffix}_640x640_nv12.hbm",
        "pose":   "yolov8n_pose_{suffix}_640x640_nv12.hbm",
        "cls":    "yolov8n_cls_{suffix}_640x640_nv12.hbm",
    },
    "yolov10": {
        "detect": "yolov10n_detect_{suffix}_640x640_nv12.hbm",
    },
    "yolo11": {
        "detect": "yolo11n_detect_{suffix}_640x640_nv12.hbm",
        "seg":    "yolo11n_seg_{suffix}_640x640_nv12.hbm",
        "pose":   "yolo11n_pose_{suffix}_640x640_nv12.hbm",
        "cls":    "yolo11n_cls_{suffix}_640x640_nv12.hbm",
    },
    "yolo12": {
        "detect": "yolo12n_detect_{suffix}_640x640_nv12.hbm",
    },
}

# Download URL base paths per yolo_type
DOWNLOAD_URL_BASE = {
    "yolov5u": "Ultralytics_YOLO_OE_3.7.0",
    "yolov8":  "Ultralytics_YOLO_OE_3.7.0",
    "yolov10": "Ultralytics_YOLO_OE_3.7.0",
    "yolo11":  "Ultralytics_YOLO_OE_3.7.0",
    "yolo12":  "Ultralytics_YOLO_OE_3.7.0",
}

def get_model_march_and_suffix(soc: str) -> tuple[str, str]:
    """Return S100 model architecture directory and filename suffix."""
    try:
        with open("/sys/class/boardinfo/board_type", "r", encoding="utf-8") as f:
            board_type = f.read().strip().lower()
    except Exception:
        board_type = soc
    if soc == "s100p" or "p" in board_type:
        return "nash-m", "nashm"
    return "nash-e", "nashe"


def get_default_model_path(task: str, model_march: str, suffix: str) -> str:
    """Get the default model path for a given yolo_type and task.

    Args:
        task: Task type (e.g., 'detect', 'seg').
        model_march: Model architecture directory (`nash-e` or `nash-m`).
        suffix: Model suffix based on SoC (`nashe` or `nashm`).

    Returns:
        Default model file path string.

    Raises:
        ValueError: If the yolo_type/task combination is not supported.
    """
    default_yolo_type = "yolo11"
    if task not in MODEL_FILE_PATTERNS[default_yolo_type]:
        raise ValueError(f"Unsupported task: {task}")

    model_file = MODEL_FILE_PATTERNS[default_yolo_type][task].format(suffix=suffix)
    return os.path.join("..", "..", "model", model_march, model_file)


def get_download_url(task: str, model_march: str, suffix: str) -> str:
    """Get the download URL for a given yolo_type and task.

    Args:
        task: Task type.
        model_march: Model architecture directory (`nash-e` or `nash-m`).
        suffix: Model suffix based on SoC.

    Returns:
        Download URL string.

    Raises:
        ValueError: If the yolo_type/task combination is not supported.
    """
    default_yolo_type = "yolo11"
    if task not in MODEL_FILE_PATTERNS[default_yolo_type]:
        raise ValueError(f"Unsupported task: {task}")

    model_file = MODEL_FILE_PATTERNS[default_yolo_type][task].format(suffix=suffix)
    url_base = DOWNLOAD_URL_BASE[default_yolo_type]
    return ("https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
            f"rdk_s100/{url_base}/{model_march}/{model_file}")


def main() -> None:
    """Run Ultralytics YOLO inference on a single image.

    This function parses command-line arguments, loads the appropriate model,
    preprocesses the input image, performs inference on the BPU, postprocesses
    results, and saves the output image.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()
    model_march, model_suffix = get_model_march_and_suffix(soc)

    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO Unified Inference")
    parser.add_argument('--task', type=str, required=True,
                        choices=['detect', 'seg', 'pose', 'cls'],
                        help="Task type: detect, seg, pose, cls")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to BPU Quantized *.hbm Model. '
                             'If not specified, uses the default path for the '
                             'selected yolo_type and task.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, '
                             'e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/bus.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label file. If not specified, uses '
                        'default COCO labels for detect/seg/pose.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection results.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold for filtering detections.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--topk', type=int, default=5,
                        help='[Cls] Top K classification results.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='[Pose] Keypoint visibility threshold.')

    opt = parser.parse_args()

    # Determine default model path if not explicitly provided
    using_default_model_path = opt.model_path is None
    if opt.model_path is None:
        opt.model_path = get_default_model_path(
            opt.task, model_march, model_suffix)

    # Download model if missing
    if not os.path.exists(opt.model_path):
        if using_default_model_path:
            download_url = get_download_url(
                opt.task, model_march, model_suffix)
            file_io.download_model_if_needed(opt.model_path, download_url)
        else:
            raise FileNotFoundError(f"Model file not found: {opt.model_path}")

    # Load image
    img = file_io.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Load labels for detection/seg/pose tasks
    labels = []
    if opt.task != 'cls':
        default_label = '../../test_data/coco_classes.names'
        label_path = opt.label_file or default_label
        if label_path and os.path.exists(label_path):
            labels = file_io.load_class_names(label_path)

    # Dispatch to the appropriate model wrapper
    result_img = None

    model_file_name = os.path.basename(opt.model_path).lower()
    if opt.task == 'detect' and model_file_name.startswith('yolov10'):
        # YOLOv10: NMS-free detection
        config = YoloV10DetectConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres
        )
        model = YoloV10Detect(config)
        model.set_scheduling_params(
            priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        boxes, scores, cls_ids = model.predict(img)
        visualize.print_detections(boxes, scores, cls_ids, labels)
        result_img = visualize.draw_boxes(
            img, boxes, cls_ids, scores, labels, visualize.rdk_colors)

    elif opt.task == 'detect':
        # DFL-based detection (v5u, v8, v11)
        config = YoloDetectConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YoloDetect(config)
        model.set_scheduling_params(
            priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        boxes, scores, cls_ids = model.predict(img)
        visualize.print_detections(boxes, scores, cls_ids, labels)
        result_img = visualize.draw_boxes(
            img, boxes, cls_ids, scores, labels, visualize.rdk_colors)

    elif opt.task == 'seg':
        # Instance segmentation
        config = YoloSegConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YoloSeg(config)
        model.set_scheduling_params(
            priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        xyxy, scores, cls_ids, masks = model.predict(img)
        visualize.draw_masks(
            img, xyxy, masks, cls_ids, visualize.rdk_colors)
        result_img = visualize.draw_boxes(
            img, xyxy, cls_ids, scores, labels, visualize.rdk_colors)

    elif opt.task == 'pose':
        # Pose estimation
        config = YoloPoseConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YoloPose(config)
        model.set_scheduling_params(
            priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        xyxy, scores, cls_ids, kpts_xy, kpts_score = model.predict(img)
        # Combine keypoint coordinates and sigmoid-activated scores for visualization
        num_instances = kpts_xy.shape[0]
        num_keypoints = kpts_xy.shape[1]
        kpts = np.zeros((num_instances, num_keypoints, 3), dtype=np.float32)
        kpts[:, :, :2] = kpts_xy
        kpts[:, :, 2] = kpts_score[:, :, 0]
        result_img = visualize.draw_pose(
            img, xyxy, kpts,
            kpt_conf_thres=opt.kpt_conf_thres,
            scores=scores, class_ids=cls_ids, colors=visualize.rdk_colors)

    elif opt.task == 'cls':
        # Classification
        config = YoloClsConfig(
            model_path=opt.model_path,
            topk=opt.topk,
            resize_type=0
        )
        model = YoloCls(config)
        model.set_scheduling_params(
            priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        results = model.predict(img)
        label_path = (opt.label_file or
                      '../../../../../datasets/imagenet/imagenet_classes.names')
        idx2label = file_io.load_labels(label_path)
        visualize.print_classification_results(results, idx2label)

    # Save result
    if result_img is not None:
        cv2.imwrite(opt.img_save_path, result_img)
        print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
