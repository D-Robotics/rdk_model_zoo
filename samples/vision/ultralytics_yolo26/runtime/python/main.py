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
It loads the appropriate model wrapper based on the `--task` argument.
"""

import argparse
import os
import sys
import logging
import cv2
import math
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath("../../../../../"))

# Import local modules
from yolo26_det import YOLO26Detect, YOLO26Config
from yolo26_seg import YOLO26Seg, YOLO26SegConfig
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

# Import utilities
import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as visualize

# Configure logger
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Unified Inference")
    parser.add_argument('--task', type=str, default='detect', choices=['detect', 'seg', 'pose', 'cls', 'obb'],
                        help="Task type: detect, seg, pose, cls, obb")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to YOLO26 *.bin Model.")
    parser.add_argument('--test-img', type=str, default='bus.jpg',
                        help='Path to Load Test Image.')
    parser.add_argument('--label-file', type=str, default=None,
                        help='Path to label file (e.g., coco.names). If None, tries to load default based on task.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to Save Result Image.')
    
    # Common Args
    parser.add_argument("--classes-num", type=int, default=80)
    parser.add_argument('--score-thres', type=float, default=0.25)
    parser.add_argument('--nms-thres', type=float, default=0.7)
    parser.add_argument("--strides", type=str, default='8,16,32')
    
    # Task-Specific Args
    parser.add_argument('--topk', type=int, default=5, help='For CLS')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='For Pose')
    parser.add_argument('--angle-sign', type=float, default=1.0, help='For OBB')
    parser.add_argument('--angle-offset', type=float, default=0.0, help='For OBB')
    parser.add_argument('--no-regularize', action='store_true', help='For OBB')

    opt = parser.parse_args()
    strides_list = [int(x) for x in opt.strides.split(',')]

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # Load Image using utils
    try:
        img = file_io.load_image(opt.test_img)
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return

    # Determine default label file if not provided
    labels = []
    if opt.label_file and os.path.exists(opt.label_file):
        if opt.task == 'cls':
            labels = file_io.load_imagenet_labels(opt.label_file)
        else:
            labels = file_io.load_class_names(opt.label_file)
    else:
        # Fallback to default paths relative to project root
        project_root = os.path.abspath("../../../../../")
        if opt.task in ['detect', 'seg', 'pose']:
            default_path = os.path.join(project_root, "datasets/coco/coco_classes.names")
            if os.path.exists(default_path):
                labels = file_io.load_class_names(default_path)
                logger.info(f"Loaded default labels from {default_path}")
        elif opt.task == 'obb':
            default_path = os.path.join(project_root, "datasets/dotav1/dota_classes.names")
            if os.path.exists(default_path):
                labels = file_io.load_class_names(default_path)
                logger.info(f"Loaded default labels from {default_path}")
        elif opt.task == 'cls':
            default_path = os.path.join(project_root, "datasets/imagenet/imagenet_classes.names")
            if os.path.exists(default_path):
                labels = file_io.load_imagenet_labels(default_path)
                logger.info(f"Loaded default labels from {default_path}")

    # Dispatcher
    if opt.task == 'detect':
        cfg = YOLO26Config(opt.model_path, opt.classes_num, opt.score_thres, opt.nms_thres, strides_list)
        model = YOLO26Detect(cfg)
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs)
        
        logger.info(f"\033[1;32mDraw Detect Results ({len(results)}): \033[0m")
        visualize.draw_detect_yolo26(img, results, labels, visualize.rdk_colors)
        cv2.imwrite(opt.img_save_path, img)
        logger.info(f"\033[1;32msaved in path: \"{opt.img_save_path}\"\033[0m")

    elif opt.task == 'seg':
        cfg = YOLO26SegConfig(opt.model_path, opt.classes_num, opt.score_thres, opt.nms_thres, strides_list)
        model = YOLO26Seg(cfg)
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs)
        
        logger.info(f"\033[1;32mDraw Seg Results ({len(results)}): \033[0m")
        visualize.draw_seg_yolo26(img, results, labels, visualize.rdk_colors)
        cv2.imwrite(opt.img_save_path, img)
        logger.info(f"\033[1;32msaved in path: \"{opt.img_save_path}\"\033[0m")

    elif opt.task == 'pose':
        cfg = YOLO26PoseConfig(opt.model_path, opt.score_thres, opt.nms_thres, opt.kpt_conf_thres, strides_list)
        model = YOLO26Pose(cfg)
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs)
        
        logger.info(f"\033[1;32mDraw Pose Results ({len(results)}): \033[0m")
        visualize.draw_pose_yolo26(img, results, visualize.COCO_SKELETON, opt.kpt_conf_thres)
        cv2.imwrite(opt.img_save_path, img)
        logger.info(f"\033[1;32msaved in path: \"{opt.img_save_path}\"\033[0m")

    elif opt.task == 'cls':
        cfg = YOLO26ClsConfig(opt.model_path, opt.topk)
        model = YOLO26Cls(cfg)
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs)
        
        logger.info(f"\033[1;32mTop-{len(results)} Results:\033[0m")
        visualize.draw_cls_yolo26(img, results, labels)

    elif opt.task == 'obb':
        cfg = YOLO26OBBConfig(opt.model_path, opt.score_thres, opt.nms_thres, 
                              opt.angle_sign, opt.angle_offset, not opt.no_regularize, strides_list)
        model = YOLO26OBB(cfg)
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs)
        
        logger.info(f"\033[1;32mDraw OBB Results ({len(results)}): \033[0m")
        visualize.draw_obb_yolo26(img, results, labels, visualize.rdk_colors)
        cv2.imwrite(opt.img_save_path, img)
        logger.info(f"\033[1;32msaved in path: \"{opt.img_save_path}\"\033[0m")

if __name__ == "__main__":
    main()