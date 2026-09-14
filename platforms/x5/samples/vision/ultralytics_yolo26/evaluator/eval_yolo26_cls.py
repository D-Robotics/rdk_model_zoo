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

"""YOLO26 ImageNet Evaluation Script.

This script evaluates the YOLO26 classification model on the ImageNet validation set,
calculating Top-1 and Top-5 accuracy.

It relies on a ground truth text file (e.g., ILSVRC2012_validation_ground_truth.txt)
where each line corresponds to the label index of an image, sorted by filename.

Typical Usage:
    python3 eval_yolo26_cls.py \
        --model-path ../model/yolo26n_cls.bin \
        --image-path /path/to/imagenet/val \
        --val-txt /path/to/val.txt
"""

import os
import cv2
import json
import argparse
import logging
import numpy as np
from time import time
import sys
from datetime import datetime

# 将 runtime 目录和项目根目录添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

# 导入推理类
from yolo26_cls import YOLO26Cls, YOLO26ClsConfig

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("YOLO26_Cls_Eval")

def main():
    parser = argparse.ArgumentParser(description="YOLO26 ImageNet Evaluation")
    parser.add_argument('--model-path', type=str, required=True, help="Path to BPU Model.") 
    parser.add_argument('--image-path', type=str, default='../../../../datasets/imagenet/val', help='Path to ImageNet val images.')
    parser.add_argument('--val-txt', type=str, required=True, help='Path to ILSVRC2012_validation_ground_truth.txt.')
    parser.add_argument('--json-save-path', type=str, default='yolo26_cls_results.json', help='Path to save results.')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to process.')
    parser.add_argument('--topk', type=int, default=5, help='Top K for evaluation.')
    opt = parser.parse_args()

    if not os.path.exists(opt.model_path):
        logger.error(f"Model not found: {opt.model_path}")
        return

    if not os.path.exists(opt.val_txt):
        logger.error(f"Ground truth file not found: {opt.val_txt}")
        return

    # 初始化推理类
    cfg = YOLO26ClsConfig(model_path=opt.model_path, topk=opt.topk)
    model = YOLO26Cls(cfg)

    # 加载 Ground Truth 列表
    val_gt = []
    with open(opt.val_txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    # 尝试取最后一个部分作为 label
                    # 注意: ImageNet 官方 GT 是 1-based (1-1000)，需要 -1 转换为 0-999
                    # 如果您使用的是 PyTorch 处理过的 0-based txt，请修改此处
                    label = int(parts[-1]) - 1
                    val_gt.append(label) 
                except ValueError:
                    pass
    logger.info(f"Loaded {len(val_gt)} labels from {opt.val_txt}")

    img_files = sorted([f for f in os.listdir(opt.image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG'))])
    if opt.limit > 0:
        img_files = img_files[:opt.limit]
    
    total_imgs = len(img_files)
    logger.info(f"Starting evaluation on {total_imgs} images...")
    
    # 检查数量是否匹配
    if len(val_gt) < total_imgs and opt.limit == 0:
        logger.warning(f"Warning: Number of images ({total_imgs}) exceeds labels ({len(val_gt)}). Some images will be skipped or mismatched if names are not aligned.")

    total_cnt, top1_cnt, top5_cnt = 0, 0, 0
    t_start = time()

    for i, img_file in enumerate(img_files):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing {i + 1}/{total_imgs} - {(i+1)/(time()-t_start):.1f} FPS")
            
        img_path = os.path.join(opt.image_path, img_file)
        
        # 获取 Ground Truth (直接按文件名排序后的索引对应)
        if i >= len(val_gt):
            break
        truth_idx = val_gt[i]

        # 推理
        try:
            img = cv2.imread(img_path)
            if img is None: continue
            nv12 = model.pre_process(img)
            outputs = model.forward(nv12)
            preds = model.post_process(outputs) # 返回 list of (class_id, score) 
        except Exception as e:
            logger.error(f"Error processing {img_file}: {e}")
            continue

        pred_ids = [p[0] for p in preds]
        
        total_cnt += 1
        if truth_idx == pred_ids[0]:
            top1_cnt += 1
            top5_cnt += 1
        elif truth_idx in pred_ids:
            top5_cnt += 1
        
    total_time = time() - t_start
    
    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": opt.model_path,
        "total_images": total_cnt,
        "top1_acc": top1_cnt / total_cnt if total_cnt > 0 else 0,
        "top5_acc": top5_cnt / total_cnt if total_cnt > 0 else 0,
        "fps": total_cnt / total_time if total_time > 0 else 0
    }

    logger.info(f"Evaluation Finished.")
    logger.info(f"Top-1 Acc: {summary['top1_acc']:.4f}")
    logger.info(f"Top-5 Acc: {summary['top5_acc']:.4f}")

    with open(opt.json_save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Results saved to {opt.json_save_path}")

if __name__ == "__main__":
    main()