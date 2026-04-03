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

"""YOLO26 COCO Segmentation Evaluation Script.

This script evaluates the YOLO26 instance segmentation model on the COCO validation set,
converting masks to polygons for COCO JSON evaluation.
"""

import os
import cv2
import json
import argparse
import logging
import numpy as np
from time import time
import sys

# 将 runtime 目录和项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

# 导入推理类
from yolo26_seg import YOLO26Seg, YOLO26SegConfig

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26_Seg_Eval")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO Segmentation Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-path', type=str, default='../../../../datasets/coco/val2017', help='Path to COCO val2017 images.')
    parser.add_argument('--ann-path', type=str, default='../../../../datasets/coco/annotations/instances_val2017.json', help='Path to COCO instances JSON.')
    parser.add_argument('--json-save-path', type=str, default='yolo26_seg_results.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 初始化推理类
    cfg = YOLO26SegConfig(
        model_path=opt.model_path,
        classes_num=opt.classes_num,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres
    )
    model = YOLO26Seg(cfg)

    # COCO 类别映射 (0-79 -> COCO IDs)
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
        try:
            image_id = int(os.path.splitext(img_file)[0])
        except:
            image_id = i

        img = cv2.imread(img_path)
        if img is None: continue
        
        nv12 = model.pre_process(img)
        outputs = model.forward(nv12)
        results = model.post_process(outputs) # list of dict {'box', 'score', 'id', 'mask'}

        for res in results:
            box = res['box'] # [x1, y1, x2, y2]
            score = res['score']
            cls_id = res['id']
            mask = res['mask'] # 概率图，位于 box 区域内

            width = box[2] - box[0]
            height = box[3] - box[1]
            if width <= 0 or height <= 0 or mask.size == 0:
                continue

            # 将 Mask 转换为多边形
            # 1. 阈值处理
            binary_mask = (mask > 0.5).astype(np.uint8)
            # 2. Resize 到检测框大小
            binary_mask = cv2.resize(binary_mask, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
            # 3. 提取轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []
            for contour in contours:
                if len(contour) < 3: continue
                # 将轮廓点偏移到原图坐标
                contour[:, :, 0] += box[0]
                contour[:, :, 1] += box[1]
                segmentation.append(contour.flatten().tolist())

            if not segmentation:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": coco_ids[cls_id],
                "bbox": [float(box[0]), float(box[1]), float(width), float(height)],
                "score": float(score),
                "segmentation": segmentation,
                "area": float(width * height) # 简化处理，通常应使用 mask 面积
            })

    total_time = time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s")
    
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

    # 评估
    if os.path.exists(opt.ann_path):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            logger.info("Evaluating Seg mAP...")
            cocoGt = COCO(opt.ann_path)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            cocoEval = COCOeval(cocoGt, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")

if __name__ == "__main__":
    main()
