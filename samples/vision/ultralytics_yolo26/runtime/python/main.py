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

"""YOLO26 Unified Inference Entry Script.

This script supports multiple YOLO26 tasks: detect, seg, pose, cls, obb.
It loads the appropriate model wrapper based on the --task argument and
executes the standard pipeline.

Example:
    python main.py --task detect --test-img ../../test_data/bus.jpg
    python main.py --task cls --test-img ../../test_data/zebra_cls.jpg
"""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath("../../../../../"))

from yolo26_det import YOLO26Detect, YOLO26DetectConfig
from yolo26_seg import YOLO26Seg, YOLO26SegConfig
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as visualize
import utils.py_utils.inspect as inspect


def main() -> None:
    """Run YOLO26 inference on a single image."""
    soc = inspect.get_soc_name().lower()
    board_type = ""
    try:
        with open("/sys/class/boardinfo/board_type", "r", encoding="utf-8") as f:
            board_type = f.read().strip().lower()
    except Exception:
        board_type = soc
    model_march = "nash-m" if soc == "s100p" or "p" in board_type else "nash-e"
    model_suffix = "nashm" if model_march == "nash-m" else "nashe"

    parser = argparse.ArgumentParser(description="YOLO26 Unified Inference")

    parser.add_argument('--task', type=str, required=True,
                        choices=['detect', 'seg', 'pose', 'cls', 'obb'],
                        help="Task type: detect, seg, pose, cls, obb")
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to BPU Quantized *.hbm Model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes.')
    parser.add_argument('--test-img', type=str, default='../../test_data/bus.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--topk', type=int, default=5,
                        help='[Cls] Top K results')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='[Pose] Keypoint visibility threshold')
    parser.add_argument('--angle-sign', type=float, default=1.0,
                        help='[OBB] Angle decoding sign multiplier')
    parser.add_argument('--angle-offset', type=float, default=0.0,
                        help='[OBB] Angle decoding offset')

    opt = parser.parse_args()

    task_model_map = {
        'detect': f'yolo26n_detect_{model_suffix}_640x640_nv12.hbm',
        'seg': f'yolo26n_seg_{model_suffix}_640x640_nv12.hbm',
        'pose': f'yolo26n_pose_{model_suffix}_640x640_nv12.hbm',
        'cls': f'yolo26n_cls_{model_suffix}_224x224_nv12.hbm',
        'obb': f'yolo26n_obb_{model_suffix}_640x640_nv12.hbm',
    }
    if opt.model_path is None:
        opt.model_path = os.path.join("..", "..", "model", model_march,
                                      task_model_map[opt.task])

    # Download model if missing.
    if not os.path.exists(opt.model_path):
        model_file = task_model_map.get(opt.task, task_model_map['detect'])
        download_url = ("https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                        f"rdk_s100/Ultralytics_YOLO_OE_3.7.0/{model_march}/{model_file}")
        file_io.download_model_if_needed(opt.model_path, download_url)

    # Load image
    img = file_io.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Load labels
    labels = []
    if opt.task != 'cls':
        default_label = '../../test_data/coco_classes.names' if opt.task != 'obb' else None
        label_path = opt.label_file or default_label
        if label_path and os.path.exists(label_path):
            labels = file_io.load_class_names(label_path)

    # Dispatch task
    result_img = None

    if opt.task == 'detect':
        config = YOLO26DetectConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YOLO26Detect(config)
        model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        boxes, scores, cls_ids = model.predict(img)
        visualize.print_detections(boxes, scores, cls_ids, labels)
        result_img = visualize.draw_boxes(img, boxes, cls_ids, scores, labels, visualize.rdk_colors)

    elif opt.task == 'seg':
        config = YOLO26SegConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YOLO26Seg(config)
        model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        xyxy, scores, cls_ids, masks = model.predict(img)
        visualize.draw_masks(img, xyxy, masks, cls_ids, visualize.rdk_colors)
        result_img = visualize.draw_boxes(img, xyxy, cls_ids, scores, labels, visualize.rdk_colors)

    elif opt.task == 'pose':
        config = YOLO26PoseConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres
        )
        model = YOLO26Pose(config)
        model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        xyxy, scores, cls_ids, kpts = model.predict(img)
        result_img = visualize.draw_pose(
            img, xyxy, kpts,
            kpt_conf_thres=opt.kpt_conf_thres,
            scores=scores, class_ids=cls_ids, colors=visualize.rdk_colors)

    elif opt.task == 'cls':
        config = YOLO26ClsConfig(
            model_path=opt.model_path,
            topk=opt.topk,
            resize_type=0
        )
        model = YOLO26Cls(config)
        model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        results = model.predict(img)
        label_path = (opt.label_file or
                      '../../../../../datasets/imagenet/imagenet_classes.names')
        idx2label = file_io.load_labels(label_path)
        visualize.print_classification_results(results, idx2label)

    elif opt.task == 'obb':
        config = YOLO26OBBConfig(
            model_path=opt.model_path,
            score_thres=opt.score_thres,
            nms_thres=opt.nms_thres,
            angle_sign=opt.angle_sign,
            angle_offset=opt.angle_offset
        )
        model = YOLO26OBB(config)
        model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
        inspect.print_model_info(model.model)

        results = model.predict(img)
        label_path = opt.label_file or '../../../../../datasets/dotav1/dota_classes.names'
        if os.path.exists(label_path):
            labels = file_io.load_class_names(label_path)
        visualize.print_obb_detections(results, labels)
        result_img = visualize.draw_obb(img, results, labels, visualize.rdk_colors)

    # Save result
    if result_img is not None:
        cv2.imwrite(opt.img_save_path, result_img)
        print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
