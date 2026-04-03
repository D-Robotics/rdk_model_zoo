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

"""YOLO26 COCO Pose Evaluation Script.

This script evaluates the YOLO26 pose estimation model on the COCO validation set,
generating a JSON file for keypoint evaluation.
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
from yolo26_pose import YOLO26Pose, YOLO26PoseConfig

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26_Pose_Eval")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO Pose Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-path', type=str, default='../../../../datasets/coco/val2017', help='Path to COCO val2017 images.')
    parser.add_argument('--ann-path', type=str, default='../../../../datasets/coco/annotations/person_keypoints_val2017.json', help='Path to COCO keypoints JSON.')
    parser.add_argument('--json-save-path', type=str, default='yolo26_pose_results.json', help='Path to save predictions.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='NMS threshold.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='Keypoint visibility threshold.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    # 初始化推理类
    cfg = YOLO26PoseConfig(
        model_path=opt.model_path,
        score_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        kpt_conf_thres=opt.kpt_conf_thres
    )
    model = YOLO26Pose(cfg)

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
        results = model.post_process(outputs) # list of dict {'box', 'score', 'kpts'}

        for res in results:
            box = res['box']
            score = res['score']
            kpts = res['kpts'] # (17, 3) -> [x, y, conf]
            
            # 转换关键点格式为 [x1, y1, v1, x2, y2, v2, ...]
            # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
            # 这里我们预测的 conf > threshold 则 v=2, 否则 v=1
            keypoints_list = []
            for k in kpts:
                x, y, conf = k
                vis = 2 if conf >= opt.kpt_conf_thres else 1
                keypoints_list.extend([float(x), float(y), int(vis)])

            width = box[2] - box[0]
            height = box[3] - box[1]
            
            predictions.append({
                "image_id": image_id,
                "category_id": 1, # COCO person is 1
                "keypoints": keypoints_list,
                "score": float(score),
                "bbox": [float(box[0]), float(box[1]), float(width), float(height)],
                "area": float(width * height)
            })

    total_time = time() - t_start
    logger.info(f"Inference finished in {total_time:.2f}s")
    
    with open(opt.json_save_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Predictions saved to {opt.json_save_path}")

    # 使用 pycocotools 评估
    if os.path.exists(opt.ann_path):
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            logger.info("Evaluating mAP...")
            cocoGt = COCO(opt.ann_path)
            cocoDt = cocoGt.loadRes(opt.json_save_path)
            cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        except Exception as e:
            logger.error(f"COCO evaluation failed: {e}")

if __name__ == "__main__":
    main()
