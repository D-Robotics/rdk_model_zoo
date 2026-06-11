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

"""YOLO26 COCO Instance Segmentation Evaluation Script.

This script evaluates the YOLO26 instance segmentation model on the COCO validation
dataset. It handles mask generation, polygon conversion, and calculates both
bounding box (BBox) and segmentation (Segm) mAP metrics.

Workflow:
    1. Load the quantized YOLO26 segmentation model (.hbm).
    2. Iterate through validation images.
    3. Run inference to get boxes and binary masks.
    4. Convert binary masks to polygons (compatible with COCO format).
    5. Save results and compute mAP using pycocotools.

Args:
    --model-path (str): Path to the BPU quantized model file (.hbm).
    --image-dir (str): Directory containing validation images.
    --annotation (str): Path to the COCO instances JSON file.
    --conf-thres (float): Confidence threshold.
    --nms-thres (float): NMS IoU threshold.
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

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo26_seg import YOLO26Seg, YOLO26SegConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalSeg")

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

def mask2polygon(mask, x_offset, y_offset):
    """Convert binary mask to polygon."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) > 2:
            cnt = cnt.flatten()
            cnt[0::2] += x_offset
            cnt[1::2] += y_offset
            polygons.append(cnt.tolist())
    return polygons

def main():
    parser = argparse.ArgumentParser(description="YOLO26 Segmentation Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-dir', type=str, default='../../../../datasets/coco/val2017', help='Directory of COCO val2017 images.')
    parser.add_argument('--annotation', type=str, default='../../../../datasets/coco/annotations/instances_val2017.json', help='COCO instances json file.')
    parser.add_argument('--json-save-path', type=str, default='results_seg.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit images.')
    parser.add_argument('--classes-num', type=int, default=80)
    
    opt = parser.parse_args()

    # 1. Init
    logger.info(f"Loading model: {opt.model_path}")
    cfg = YOLO26SegConfig(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Seg(cfg)

    # 2. Images
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

    # 3. Inference
    for i, img_file in tqdm(enumerate(img_files), total=total_imgs, desc="Segment"):
        img_path = os.path.join(opt.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # Returns: boxes (N,4), scores (N,), cls_ids (N,), masks (list of N arrays)
        xyxy, scores, cls_ids, masks = model.predict(img)

        try:
            image_id = int(os.path.splitext(img_file)[0])
        except:
            image_id = i

        for box, score, cid, mask in zip(xyxy, scores, cls_ids, masks):
            if mask is None or mask.size == 0: continue
            
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            # Map ID
            if cid < len(COCO_IDS):
                category_id = COCO_IDS[cid]
            else:
                continue

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            seg_polys = mask2polygon(mask.astype(np.uint8), x1, y1)
            
            if seg_polys:
                predictions.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "segmentation": seg_polys
                })

    # 4. Save & Eval
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Inference finished in {time.time() - t_start:.2f}s")
    logger.info(f"Predictions saved to {opt.json_save_path}")

    if os.path.exists(opt.annotation):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            cocoGt = COCO(opt.annotation)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            
            # Evaluate Box
            logger.info("Evaluating BBox...")
            cocoEvalBox = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEvalBox.evaluate()
            cocoEvalBox.accumulate()
            cocoEvalBox.summarize()

            # Evaluate Mask
            logger.info("Evaluating Segm...")
            cocoEvalSeg = COCOeval(cocoGt, cocoDt, 'segm')
            cocoEvalSeg.evaluate()
            cocoEvalSeg.accumulate()
            cocoEvalSeg.summarize()
        except Exception as e:
            logger.error(f"COCO eval failed: {e}")

if __name__ == "__main__":
    main()
