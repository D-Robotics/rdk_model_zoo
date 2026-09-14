import os
import sys
import json
import time
import argparse

import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo_cls import YoloCls, YoloClsConfig


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Classification Evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--val-txt", required=True)
    parser.add_argument("--json-save-path", default="results_cls.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--label-offset", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1000)
    args = parser.parse_args()

    model = YoloCls(YoloClsConfig(model_path=args.model_path, topk=args.topk))
    gt_map = {}
    with open(args.val_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                gt_map[os.path.basename(parts[0])] = int(parts[1]) + args.label_offset

    img_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".jpeg"))])
    if args.limit > 0:
        img_files = img_files[:args.limit]

    top1 = 0
    top5 = 0
    total = 0
    start = time.time()
    for img_file in img_files:
        if img_file not in gt_map:
            continue
        img = cv2.imread(os.path.join(args.image_dir, img_file))
        if img is None:
            continue
        preds = model(img, topk=args.topk)
        pred_ids = [p[0] for p in preds]
        gt = gt_map[img_file]
        total += 1
        if pred_ids and pred_ids[0] == gt:
            top1 += 1
        if gt in pred_ids[:5]:
            top5 += 1
        if args.log_interval > 0 and total % args.log_interval == 0:
            elapsed = time.time() - start
            print({
                "progress": total,
                "top1": top1 / total if total else 0.0,
                "top5": top5 / total if total else 0.0,
                "elapsed_sec": elapsed,
            }, flush=True)

    summary = {
        "total": total,
        "top1": top1 / total if total else 0.0,
        "top5": top5 / total if total else 0.0,
        "elapsed_sec": time.time() - start,
    }
    with open(args.json_save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
