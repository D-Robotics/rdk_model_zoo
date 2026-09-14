import os
import sys
import json
import time
import argparse

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo_pose import YoloPose, YoloPoseConfig


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Pose Evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.7)
    parser.add_argument("--json-save-path", default="results_pose.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    coco = COCO(args.annotation)
    image_ids = coco.getImgIds()
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    model = YoloPose(YoloPoseConfig(model_path=args.model_path))
    results = []
    start = time.time()
    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        img = cv2.imread(os.path.join(args.image_dir, info["file_name"]))
        if img is None:
            continue
        boxes, scores, cls_ids, kpts_xy, kpts_score = model(img, score_thres=args.conf_thres, nms_thres=args.nms_thres)
        for box, score, kxy, kscore in zip(boxes, scores, kpts_xy, kpts_score):
            x1, y1, x2, y2 = [float(v) for v in box]
            flat_kpts = []
            for (x, y), ks in zip(kxy, kscore):
                ks_value = ks.item() if hasattr(ks, "item") else float(ks)
                flat_kpts.extend([float(x), float(y), 1 if ks_value > 0 else 0])
            results.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "keypoints": flat_kpts,
            })

    with open(args.json_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(args.json_save_path)
    coco_eval = COCOeval(coco, coco_dt, "keypoints")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")


if __name__ == "__main__":
    main()

