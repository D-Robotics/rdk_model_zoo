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

# flake8: noqa: E501
# flake8: noqa: E402

"""YOLO26 COCO Object Detection Evaluation Script.

This script evaluates the YOLO26 object detection model on the COCO validation
dataset. It runs inference on the RDK BPU, generates predictions in COCO JSON
format, and calculates standard mAP metrics using pycocotools.

Workflow:
    1. Load the quantized YOLO26 detection model (.hbm).
    2. Iterate through images in the validation directory.
    3. Perform inference using the unified `YOLO26Detect` wrapper.
    4. Save detection results to a JSON file.
    5. Compute mAP (0.5:0.95) using ground truth annotations.

Args:
    --model-path (str): Path to the BPU quantized model file (.hbm).
    --image-dir (str): Directory containing validation images.
    --annotation (str): Path to the COCO instances JSON file.
    --conf-thres (float): Confidence threshold for filtering predictions.
    --nms-thres (float): IoU threshold for Non-Maximum Suppression.
"""

import os
import sys
import json
import time
import argparse
import logging
import cv2
import numpy as np
from tqdm import tqdm

# Add runtime directory and project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo26_det import YOLO26Detect, YOLO26Config
import utils.py_utils.file_io as file_io

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalDet")

# Mapping from model training indices (0-79) to COCO original category IDs (1-90).
# COCO 2017 dataset has 80 valid classes, but IDs span from 1 to 90 with some gaps
# (e.g., ID 12, 26, 29... are missing). Evaluation tools (pycocotools) require
# original IDs to match ground truth.
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-dir', type=str, default='../../../../datasets/coco/val2017', help='Directory of COCO val2017 images.')
    parser.add_argument('--annotation', type=str, default='../../../../datasets/coco/annotations/instances_val2017.json', help='COCO instances json file.')
    parser.add_argument('--json-save-path', type=str, default='results_det.json', help='Path to save result JSON.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold (low for mAP evaluation).')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS IoU threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit images (0 for all).')
    
    # Model config args
    parser.add_argument("--classes-num", type=int, default=80)
    parser.add_argument("--strides", type=str, default='8,16,32')

    opt = parser.parse_args()
    strides_list = [int(x) for x in opt.strides.split(',')]

    # 1. Initialize Model
    logger.info(f"Loading model: {opt.model_path}")
    cfg = YOLO26Config(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        strides=strides_list
    )
    model = YOLO26Detect(cfg)

    # 2. Prepare Images
    if not os.path.exists(opt.image_dir):
        logger.error(f"Image directory not found: {opt.image_dir}")
        return

    img_files = sorted([f for f in os.listdir(opt.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if opt.limit > 0:
        img_files = img_files[:opt.limit]
    
    total_imgs = len(img_files)
    logger.info(f"Evaluating on {total_imgs} images...")

    predictions = []
    t_start = time.time()

    # 3. Inference Loop
    for i, img_file in tqdm(enumerate(img_files), total=total_imgs, desc="Detect"):
        img_path = os.path.join(opt.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # Use unified predict() method
        boxes, scores, cls_ids = model.predict(img)

        # 4. Format Results
        try:
            image_id = int(os.path.splitext(img_file)[0])
        except ValueError:
            image_id = i 

        for box, score, cid in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Map model class ID (0-79) to COCO category ID
            if cid < len(COCO_IDS):
                category_id = COCO_IDS[cid]
            else:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(score)
            })

    total_time = time.time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s ({total_imgs/total_time:.1f} FPS)")

    # 5. Save JSON
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Results saved to {opt.json_save_path}")

    # 6. Calculate mAP
    if os.path.exists(opt.annotation):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            
            cocoGt = COCO(opt.annotation)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        except ImportError:
            logger.warning("pycocotools not installed. Skipping mAP calculation.")
    else:
        logger.info("Annotation file not provided. Skipping mAP calculation.")

if __name__ == "__main__":
    main()