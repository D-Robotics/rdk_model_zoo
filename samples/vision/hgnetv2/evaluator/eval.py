#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
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

"""HGNetV2 ImageNet Evaluation Script (recursive directories, CSV ground truth).

Supports datasets with subfolders. The CSV file should contain relative paths
(e.g., 'imagenet_val/n01440764/xxx.JPEG') and zero‑based labels.
The script recursively scans `--image-path`, computes the relative path of each image,
and matches it against the CSV entries exactly.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from time import time

import cv2

# 添加项目根目录与 runtime 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
sys.path.append(project_root)
sys.path.append(runtime_path)

from hgnetv2 import HGNetV2, HGNetV2Config

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("HGNetV2_Eval_Recursive")


def load_ground_truth_csv(csv_path: str):
    """
    从 CSV 文件加载 ground truth，返回 {相对路径字符串: 标签} 的字典。
    第一列保持原样（不做任何 basename 提取），第二列为标签（0‑based）。
    自动跳过标题行（如果第一行包含 'image' 和 'category'）。
    """
    gt_map = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            # 跳过可能的标题行
            if row[0].strip().lower() == "image:file" and row[1].strip().lower() == "category":
                continue
            img_rel_path = row[0].strip()
            try:
                label = int(row[1].strip())
            except ValueError:
                logger.warning(f"Invalid label in CSV: {row[1]}, skipping")
                continue

            # 统一路径分隔符为 '/'，便于跨平台匹配（Windows 下 CSV 中可能也是 '/'）
            img_rel_path = img_rel_path.replace("\\", "/")
            gt_map[img_rel_path] = label

    logger.info(f"Loaded {len(gt_map)} ground truth entries from {csv_path}")
    return gt_map


def collect_images_with_relative_paths(image_root: str):
    """
    递归遍历 image_root 目录，收集所有图像文件。
    返回列表，每个元素为 (相对路径字符串, 绝对路径)。
    相对路径使用 '/' 作为分隔符，并且不包含前导 './'。
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".JPEG")
    results = []
    # 确保根目录以 os.sep 结尾，方便后续计算相对路径
    root_norm = os.path.abspath(image_root) + os.sep

    for dirpath, _, filenames in os.walk(image_root):
        for f in filenames:
            if f.lower().endswith(image_extensions):
                abs_path = os.path.join(dirpath, f)
                # 计算相对于 image_root 的相对路径
                rel_path = os.path.relpath(abs_path, image_root)
                # 统一分隔符为 '/'
                rel_path = rel_path.replace("\\", "/")
                results.append((rel_path, abs_path))

    # 可选：排序，便于观察
    results.sort(key=lambda x: x[0])
    logger.info(f"Found {len(results)} images under {image_root}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HGNetV2 ImageNet Evaluation with recursive subdirectories and CSV ground truth")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the quantized HGNetV2 *.bin model.")
    parser.add_argument("--image-path", type=str, required=True, help="Root directory containing the validation images (with subfolders).")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to the CSV file with columns: image_path, category (0‑based).")
    parser.add_argument("--label-file", type=str, default="", help="Path to ImageNet class names (optional).")
    parser.add_argument("--json-save-path", type=str, default="hgnetv2_cls_results.json", help="Path to save evaluation results.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of images to evaluate (0 = all).")
    parser.add_argument("--topk", type=int, default=5, help="Top K for accuracy evaluation.")
    parser.add_argument("--resize-type", type=int, default=0, help="Resize type (0: direct, 1: letterbox).")
    parser.add_argument("--priority", type=int, default=0, help="Model scheduling priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indices.")
    args = parser.parse_args()

    # 检查输入
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        return
    if not os.path.exists(args.val_csv):
        logger.error(f"CSV file not found: {args.val_csv}")
        return
    if not os.path.isdir(args.image_path):
        logger.error(f"Image directory not found: {args.image_path}")
        return

    # 加载 ground truth 映射 (相对路径 -> 标签)
    gt_map = load_ground_truth_csv(args.val_csv)

    # 初始化模型
    config = HGNetV2Config(
        model_path=args.model_path,
        label_file=args.label_file if args.label_file else "",
        resize_type=args.resize_type,
        topk=args.topk,
    )
    model = HGNetV2(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    # 收集所有图像及其相对路径
    images = collect_images_with_relative_paths(args.image_path)
    if args.limit > 0:
        images = images[:args.limit]

    total_imgs = len(images)
    logger.info(f"Will evaluate up to {total_imgs} images")

    # 统计
    matched = 0
    total_cnt = 0
    top1_cnt = 0
    top5_cnt = 0
    t_start = time()

    for idx, (rel_path, abs_path) in enumerate(images):
        if (idx + 1) % 100 == 0:
            fps = (idx + 1) / (time() - t_start)
            logger.info(f"Processed {idx + 1}/{total_imgs} - {fps:.1f} FPS")

        # 查找 ground truth
        truth_label = gt_map.get(rel_path)
        if truth_label is None:
            logger.debug(f"No ground truth for {rel_path}, skipping")
            continue
        matched += 1

        img = cv2.imread(abs_path)
        if img is None:
            logger.error(f"Failed to read image: {abs_path}")
            continue

        try:
            topk_idx, topk_prob, _ = model.predict(img)
            pred_ids = topk_idx.tolist()
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            continue

        total_cnt += 1
        if truth_label == pred_ids[0]:
            top1_cnt += 1
            top5_cnt += 1
        elif truth_label in pred_ids:
            top5_cnt += 1

    elapsed = time() - t_start
    top1_acc = top1_cnt / total_cnt if total_cnt else 0.0
    top5_acc = top5_cnt / total_cnt if total_cnt else 0.0
    fps = total_cnt / elapsed if elapsed else 0.0

    summary = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model_path,
        "image_root": args.image_path,
        "csv_file": args.val_csv,
        "total_images_scanned": total_imgs,
        "matched_to_gt": matched,
        "successful_inferences": total_cnt,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "fps": fps,
        "config": {
            "resize_type": args.resize_type,
            "topk": args.topk,
            "bpu_cores": args.bpu_cores,
            "priority": args.priority,
        },
    }

    logger.info("Evaluation finished.")
    logger.info(f"Matched {matched}/{total_imgs} images to ground truth")
    logger.info(f"Successful inferences: {total_cnt}")
    logger.info(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_cnt}/{total_cnt})")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_cnt}/{total_cnt})")
    logger.info(f"Average FPS: {fps:.2f}")

    with open(args.json_save_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Results saved to {args.json_save_path}")


if __name__ == "__main__":
    main()