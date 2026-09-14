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

"""YOLO26 COCO Pose Estimation Evaluation Script.

This script evaluates the YOLO26 pose estimation model on the COCO validation
dataset. It predicts human keypoints and calculates OKS-based mAP metrics.

Workflow:
    1. Load the quantized YOLO26 pose model (.hbm).
    2. Iterate through validation images.
    3. Decode bounding boxes and 17 keypoints (x, y, conf).
    4. Format results (including visibility flags) for COCO evaluation.
    5. Compute Keypoint mAP using pycocotools.

Args:
    --model-path (str): Path to the BPU quantized model file (.hbm).
    --image-dir (str): Directory containing validation images.
    --annotation (str): Path to the COCO person keypoints JSON file.
    --kpt-conf-thres (float): Threshold to mark a keypoint as visible.
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

from yolo26_pose import YOLO26Pose, YOLO26PoseConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalPose")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO Pose Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-dir', type=str, default='../../../../datasets/coco/val2017', help='Directory of COCO val2017 images.')
    parser.add_argument('--annotation', type=str, default='../../../../datasets/coco/annotations/person_keypoints_val2017.json', help='Path to COCO keypoints JSON.')
    parser.add_argument('--json-save-path', type=str, default='results_pose.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit images.')
    opt = parser.parse_args()

    # 1. Init
    logger.info(f"Loading model: {opt.model_path}")
    cfg = YOLO26PoseConfig(
        model_path=opt.model_path,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Pose(cfg)

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
    for i, img_file in tqdm(enumerate(img_files), total=total_imgs, desc="Pose"):
        img_path = os.path.join(opt.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Predict
        # returns xyxy, scores, cls_ids, kpts
        xyxy, scores, cls_ids, kpts = model.predict(img)

        try:
            image_id = int(os.path.splitext(img_file)[0])
        except:
            image_id = i

        for box, score, kpt in zip(xyxy, scores, kpts):
            # kpt shape (17, 3) -> x, y, conf
            
            # Format keypoints: [x1, y1, v1, ...]
            # v=2 (visible) if conf > 0.5 (arbitrary), else v=1 (labeled but not visible)
            # COCO usually just needs v=1 if predicted.
            flat_kpts = []
            for k in kpt:
                flat_kpts.extend([float(k[0]), float(k[1]), 1]) # v=1

            width = box[2] - box[0]
            height = box[3] - box[1]
            
            predictions.append({
                "image_id": image_id,
                # COCO Category ID for 'person' is 1
                "category_id": 1,
                "keypoints": flat_kpts,
                "score": float(score),
                "bbox": [float(box[0]), float(box[1]), float(width), float(height)],
                "area": float(width * height)
            })

    total_time = time.time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s")
    
    # 4. Save & Eval
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

    if os.path.exists(opt.annotation):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            cocoGt = COCO(opt.annotation)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            
            logger.info("Evaluating Keypoints mAP...")
            cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")

if __name__ == "__main__":
    main()