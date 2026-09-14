#!/usr/bin/env python3
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

"""YOLO26 COCO Evaluation Script for RDK X5.

This script runs inference using the YOLO26 detection model on the COCO validation dataset
and calculates the mean Average Precision (mAP) metrics using pycocotools.

Typical Usage:
    python3 eval_yolo26_det.py --model-path ../model/yolo26n_bpu.bin --limit 100
"""

import os
import cv2
import json
import argparse
import logging
import numpy as np
from time import time
import sys

# Add runtime directory and project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

# Import the optimized detector class
from yolo26_det import YOLO26Detect, YOLO26Config

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO_Eval")

def main():
    """Run COCO evaluation."""
    parser = argparse.ArgumentParser(description="YOLO26 COCO Evaluation")
    parser.add_argument('--model-path', type=str, default='yolo26n_bpu_bayese_640x640_nv12.bin', help="Path to BPU Model.") 
    parser.add_argument('--image-path', type=str, default='../../../../datasets/coco/val2017', help='Path to COCO val2017 images.')
    parser.add_argument('--ann-path', type=str, default='../../../../datasets/coco/annotations/instances_val2017.json', help='Path to COCO instances_val2017.json (optional, for direct mAP calculation).')
    parser.add_argument('--json-save-path', type=str, default='yolo26_predictions.json', help='Path to save prediction JSON.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold (low for mAP evaluation).')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS IoU threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to process (0 for all).')
    parser.add_argument('--classes-num', type=int, default=80, help="Classes Num to Detect.")
    parser.add_argument("--strides", type=lambda s: list(map(int, s.split(','))), default=[8, 16, 32], help="--strides 8, 16, 32")
    opt = parser.parse_args()
    logger.info(opt)

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    if not os.path.exists(opt.image_path):
        logger.error(f"Image path not found: {opt.image_path}")
        return

    # Initialize Detector
    # Note: Using a low score_thres is crucial for high mAP
    cfg = YOLO26Config(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        strides=opt.strides
    )
    detector = YOLO26Detect(cfg)

    # COCO Category ID Mapping (0-79 -> COCO IDs)
    # COCO IDs are not continuous (missing 12, 26, etc.)
    coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    img_files = sorted([f for f in os.listdir(opt.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if opt.limit > 0:
        img_files = img_files[:opt.limit]
    
    total_imgs = len(img_files)
    logger.info(f"Starting evaluation on {total_imgs} images...")
    
    predictions = []
    t_start = time()

    for i, img_file in enumerate(img_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i + 1}/{total_imgs} - {(i+1)/(time()-t_start):.1f} FPS")
            
        img_path = os.path.join(opt.image_path, img_file)
        
        # 1. Load
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 2. Inference
        nv12_data = detector.pre_process(img)
        outputs = detector.forward(nv12_data)
        results = detector.post_process(outputs)
        
        # 3. Format for COCO
        # Extract image ID from filename (e.g., 000000123456.jpg -> 123456)
        try:
            image_id = int(os.path.splitext(img_file)[0])
        except ValueError:
            # If filename isn't an ID, use index (warning: might not match ground truth)
            image_id = i 

        for cls_id, score, x1, y1, x2, y2 in results:
            width = x2 - x1
            height = y2 - y1
            
            prediction = {
                "image_id": image_id,
                "category_id": coco_ids[cls_id],
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(score)
            }
            predictions.append(prediction)

    total_time = time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s ({total_imgs/total_time:.1f} FPS)")
    
    # Save JSON
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

    # Calculate mAP using pycocotools if available and annotation file exists
    if os.path.exists(opt.ann_path):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            
            logger.info("Evaluating mAP...")
            cocoGt = COCO(opt.ann_path)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
            # Print detailed mAP metrics
            print(f"AP @ IoU=0.50:0.95 (area=all):    {cocoEval.stats[0]:.4f}")
            print(f"AP @ IoU=0.50      (area=all):    {cocoEval.stats[1]:.4f}")
            print(f"AP @ IoU=0.75      (area=all):    {cocoEval.stats[2]:.4f}")
            print(f"AP @ IoU=0.50:0.95 (area=small):  {cocoEval.stats[3]:.4f}")
            print(f"AP @ IoU=0.50:0.95 (area=medium): {cocoEval.stats[4]:.4f}")
            print(f"AP @ IoU=0.50:0.95 (area=large):  {cocoEval.stats[5]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=all, maxDets=1):   {cocoEval.stats[6]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=all, maxDets=10):  {cocoEval.stats[7]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=all, maxDets=100): {cocoEval.stats[8]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=small):  {cocoEval.stats[9]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=medium): {cocoEval.stats[10]:.4f}")
            print(f"AR @ IoU=0.50:0.95 (area=large):  {cocoEval.stats[11]:.4f}")
            
        except ImportError:
            logger.warning("pycocotools not found. Please install it to calculate mAP automatically: pip install pycocotools")
        except Exception as e:
            logger.error(f"Error during COCO evaluation: {e}")
    else:
        logger.info(f"Annotation file not found at {opt.ann_path}. Skipping mAP calculation. Use the saved JSON with an external eval script.")

if __name__ == "__main__":
    main()
