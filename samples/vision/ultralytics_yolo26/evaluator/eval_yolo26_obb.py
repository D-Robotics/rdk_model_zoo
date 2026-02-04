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

"""YOLO26 OBB (Oriented Bounding Box) Evaluation Script.

This script evaluates the YOLO26 OBB model, typically used for datasets like DOTA.
It generates predictions with 8-point polygon coordinates representing rotated boxes.
It also supports reading ground truth labels in YOLO OBB format (normalized x1 y1 ...).
"""

import os
import cv2
import json
import argparse
import logging
import numpy as np
import math
from time import time
import sys

# 将 runtime 目录和项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

# 导入推理类
from yolo26_obb import YOLO26OBB, YOLO26OBBConfig

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26_OBB_Eval")

def rbox2poly(obbox):
    """
    Convert oriented bounding box to polygon.
    Args:
        obbox (list): [cx, cy, w, h, angle_rad]
    Returns:
        list: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    c, s = math.cos(obbox[4]), math.sin(obbox[4])
    w2 = obbox[2] / 2
    h2 = obbox[3] / 2
    
    # Corners relative to center
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    poly = []
    
    for dx, dy in corners:
        x = obbox[0] + dx * c - dy * s
        y = obbox[1] + dx * s + dy * c
        poly.append(float(x))
        poly.append(float(y))
        
    return poly

def main():
    parser = argparse.ArgumentParser(description="YOLO26 OBB Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-path', type=str, default='../../../../datasets/dotav1/val/images', help='Path to DOTA/OBB val images.')
    parser.add_argument('--label-path', type=str, default=None, help='Path to YOLO OBB format labels (.txt).')
    parser.add_argument('--json-save-path', type=str, default='yolo26_obb_results.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images.')
    parser.add_argument("--strides", type=lambda s: list(map(int, s.split(','))), default=[8, 16, 32], help="--strides 8, 16, 32")
    parser.add_argument('--angle-sign', type=float, default=1.0, help='Angle sign for decoding.')
    parser.add_argument('--angle-offset', type=float, default=0.0, help='Angle offset in degrees.')
    
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 初始化推理类
    cfg = YOLO26OBBConfig(
        model_path=opt.model_path,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        angle_sign=opt.angle_sign,
        angle_offset=opt.angle_offset,
        strides=opt.strides
    )
    model = YOLO26OBB(cfg)

    if not os.path.exists(opt.image_path):
        logger.error(f"Image path not found: {opt.image_path}")
        return

    img_files = sorted([f for f in os.listdir(opt.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
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
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 如果提供了 label-path，可以在此处读取 GT（目前仅做解析演示）
        if opt.label_path:
            label_file = os.path.join(opt.label_path, os.path.splitext(img_file)[0] + ".txt")
            if os.path.exists(label_file):
                # 这里可以添加加载 GT 并与预测结果对比的逻辑
                pass

        try:
            fname = os.path.splitext(img_file)[0]
            if fname.startswith('P'):
                image_id = int(fname[1:])
            else:
                image_id = i
        except:
            image_id = i

        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs) # list of dict {'rrect', 'score', 'id'}

        for res in results:
            rrect = res['rrect'] # (cx, cy, w, h, angle_rad)
            poly = rbox2poly(rrect)
            
            # 计算外接水平框
            xs = poly[0::2]
            ys = poly[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            predictions.append({
                "image_id": image_id,
                "category_id": int(res['id']),
                "score": float(res['score']),
                "segmentation": [poly],
                "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)],
                "area": float(rrect[2] * rrect[3]),
                "iscrowd": 0
            })

    total_time = time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s")
    
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

if __name__ == "__main__":
    main()