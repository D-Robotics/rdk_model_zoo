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

"""YOLO26 ImageNet Classification Evaluation Script.

This script evaluates the YOLO26 classification model on the ImageNet validation
set. It compares model predictions against a ground truth file.

Workflow:
    1. Load the quantized YOLO26 classification model (.hbm).
    2. Load ground truth mapping (filename -> label) from a text file.
    3. Iterate through validation images and run inference.
    4. Calculate Top-1 and Top-5 accuracy and FPS.
    5. Save a summary JSON report.

Args:
    --model-path (str): Path to the BPU quantized model file (.hbm).
    --image-dir (str): Directory containing validation images.
    --val-txt (str): Path to the ground truth file (format: "filename label").
    --topk (int): Number of top classes to consider for evaluation.
"""

import os
import sys
import json
import time
import argparse
import logging
import cv2
from datetime import datetime
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo26_cls import YOLO26Cls, YOLO26ClsConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalCls")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 ImageNet Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-dir', type=str, default='../../../../datasets/imagenet/val', help='Directory of ImageNet val images.')
    parser.add_argument('--val-txt', type=str, required=True, help='Path to val.txt (format: "ILSVRC2012_val_00000001.JPEG 65").')
    parser.add_argument('--json-save-path', type=str, default='results_cls.json', help='Path to save summary.')
    parser.add_argument('--limit', type=int, default=0, help='Limit images.')
    parser.add_argument('--topk', type=int, default=5, help='Top K to evaluate.')
    parser.add_argument('--label-offset', type=int, default=0, help='Offset to add to labels in txt (use -1 if txt is 1-based).')
    opt = parser.parse_args()

    # 1. Init Model
    logger.info(f"Loading model: {opt.model_path}")
    cfg = YOLO26ClsConfig(model_path=opt.model_path, topk=opt.topk)
    model = YOLO26Cls(cfg)

    # 2. Load GT Mapping
    if not os.path.exists(opt.val_txt):
        logger.error(f"GT file not found: {opt.val_txt}")
        return
    
    gt_map = {}
    with open(opt.val_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Format: filename label (e.g. ILSVRC2012_val_00000001.JPEG 65)
                fname = os.path.basename(parts[0])
                label = int(parts[1]) + opt.label_offset
                gt_map[fname] = label
            elif len(parts) == 1:
                # Fallback for label-only files (assumes sequential order)
                idx = len(gt_map)
                gt_map[idx] = int(parts[0]) + opt.label_offset
    
    logger.info(f"Loaded {len(gt_map)} ground truth labels.")

    # 3. Process Images
    if not os.path.exists(opt.image_dir):
        logger.error(f"Image directory not found: {opt.image_dir}")
        return

    img_files = sorted([f for f in os.listdir(opt.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
    if opt.limit > 0:
        img_files = img_files[:opt.limit]
    
    total_imgs = len(img_files)
    logger.info(f"Evaluating on {total_imgs} images...")

    total_cnt, top1_cnt, top5_cnt = 0, 0, 0
    t_start = time.time()

    # 4. Inference Loop
    for i, img_file in tqdm(enumerate(img_files), total=total_imgs, desc="Classify"):
        # Get Ground Truth
        if img_file in gt_map:
            truth_idx = gt_map[img_file]
        elif i in gt_map:
            truth_idx = gt_map[i]
        else:
            continue
            
        img_path = os.path.join(opt.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # Unified predict() returns list of (class_id, score)
        results = model.predict(img)
        pred_ids = [r[0] for r in results]
        
        total_cnt += 1
        if pred_ids:
            if truth_idx == pred_ids[0]:
                top1_cnt += 1
            if truth_idx in pred_ids[:5]:
                top5_cnt += 1

    total_time = time.time() - t_start
    fps = total_cnt / total_time if total_time > 0 else 0
    
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": opt.model_path,
        "total": total_cnt,
        "top1": top1_cnt / total_cnt if total_cnt else 0,
        "top5": top5_cnt / total_cnt if total_cnt else 0,
        "fps": fps
    }

    print(f"\n{'='*20} Results {'='*20}")
    print(f"Top-1 Acc: {summary['top1']:.4f}")
    print(f"Top-5 Acc: {summary['top5']:.4f}")
    print(f"FPS:       {summary['fps']:.2f}")
    print(f"{ '='*49}\n")

    with open(opt.json_save_path, 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()