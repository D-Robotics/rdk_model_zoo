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

"""YOLO26 OBB (Oriented Bounding Box) Evaluation Script.

This script evaluates the YOLO26 OBB model, typically used for aerial datasets like
DOTA. It generates predictions with 8-point polygon coordinates, which can be used
with DOTA evaluation tools.

Workflow:
    1. Load the quantized YOLO26 OBB model (.hbm).
    2. Iterate through validation images.
    3. Run inference to get rotated boxes (cx, cy, w, h, angle).
    4. Convert rotated boxes to 8-point polygons [x1, y1, ..., x4, y4].
    5. Save predictions to a JSON file (standard format) for offline evaluation.

Args:
    --model-path (str): Path to the BPU quantized model file (.hbm).
    --image-dir (str): Directory containing validation images.
    --angle-sign (float): Multiplier for angle decoding (1.0 or -1.0).
    --angle-offset (float): Angle offset in degrees.
"""

import os
import sys
import json
import time
import argparse
import logging
import cv2
import math
import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("EvalOBB")

def rbox2poly(rrect):
    """Convert (cx, cy, w, h, angle) to 8-point polygon [x1, y1, ...]."""
    cx, cy, w, h, a = rrect
    c, s = math.cos(a), math.sin(a)
    w2, h2 = w / 2, h / 2
    
    # Corners in local frame (unrotated)
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    poly = []
    
    for dx, dy in corners:
        x = cx + dx * c - dy * s
        y = cy + dx * s + dy * c
        poly.append(float(x))
        poly.append(float(y))
        
    return poly

def main():
    parser = argparse.ArgumentParser(description="YOLO26 OBB Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-dir', type=str, default='../../../../datasets/dotav1/val/images', help='Directory of val images.')
    parser.add_argument('--json-save-path', type=str, default='results_obb.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit images.')
    
    # OBB specific
    parser.add_argument('--angle-sign', type=float, default=1.0, help='Angle sign.')
    parser.add_argument('--angle-offset', type=float, default=0.0, help='Angle offset.')
    parser.add_argument("--strides", type=str, default='8,16,32')
    
    opt = parser.parse_args()
    strides_list = [int(x) for x in opt.strides.split(',')]

    # 1. Init
    logger.info(f"Loading model: {opt.model_path}")
    cfg = YOLO26OBBConfig(
        model_path=opt.model_path,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        angle_sign=opt.angle_sign,
        angle_offset=opt.angle_offset,
        strides=strides_list
    )
    model = YOLO26OBB(cfg)

    # 2. Images
    if not os.path.exists(opt.image_dir):
        logger.error(f"Image directory not found: {opt.image_dir}")
        return

    img_files = sorted([f for f in os.listdir(opt.image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    if opt.limit > 0:
        img_files = img_files[:opt.limit]
    
    total_imgs = len(img_files)
    logger.info(f"Evaluating on {total_imgs} images...")
    
    predictions = []
    t_start = time.time()

    # 3. Inference
    for i, img_file in tqdm(enumerate(img_files), total=total_imgs, desc="OBB"):
        img_path = os.path.join(opt.image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        # Unified predict()
        # returns list of dicts: {'rrect': (cx, cy, w, h, a), 'score': s, 'id': id}
        results = model.predict(img)

        try:
            fname = os.path.splitext(img_file)[0]
            # Try parsing typical DOTA numeric filename 'P1234' -> 1234
            if fname.startswith('P') and fname[1:].isdigit():
                image_id = int(fname[1:])
            else:
                image_id = i
        except:
            image_id = i

        for res in results:
            rrect = res['rrect']
            poly = rbox2poly(rrect)
            
            # Compute Axis-Aligned BBox (AABB) for standard COCO evaluators
            xs = poly[0::2]
            ys = poly[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            predictions.append({
                "image_id": image_id,
                "category_id": int(res['id']),
                "score": float(res['score']),
                "segmentation": [poly], # Polygon format [x1, y1, ...]
                "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                "area": float(rrect[2] * rrect[3]),
                "iscrowd": 0
            })

    total_time = time.time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s")
    
    # 4. Save
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

if __name__ == "__main__":
    main()
