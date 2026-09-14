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

from yolo_detect import YoloDetect, YoloDetectConfig
from yolo_v10detect import YoloV10Detect, YoloV10DetectConfig


def build_model(model_path: str):
    name = os.path.basename(model_path).lower()
    if "yolov10" in name:
        return YoloV10Detect(YoloV10DetectConfig(model_path=model_path)), True
    return YoloDetect(YoloDetectConfig(model_path=model_path)), False


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Detection Evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.7)
    parser.add_argument("--json-save-path", default="results_det.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    coco = COCO(args.annotation)
    image_ids = coco.getImgIds()
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    coco_cat_ids = [cat["id"] for cat in coco.dataset["categories"]]

    model, is_yolov10 = build_model(args.model_path)
    results = []
    start = time.time()

    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        path = os.path.join(args.image_dir, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            continue
        if is_yolov10:
            boxes, scores, cls_ids = model(img, score_thres=args.conf_thres)
        else:
            boxes, scores, cls_ids = model(img, score_thres=args.conf_thres, nms_thres=args.nms_thres)
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = [float(v) for v in box]
            results.append({
                "image_id": image_id,
                "category_id": int(coco_cat_ids[int(cls_id)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    with open(args.json_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(args.json_save_path)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")


if __name__ == "__main__":
    main()


