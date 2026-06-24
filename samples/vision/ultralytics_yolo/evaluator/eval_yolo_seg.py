import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

current_dir = os.path.dirname(os.path.abspath(__file__))
runtime_path = os.path.abspath(os.path.join(current_dir, "../runtime/python"))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
sys.path.append(runtime_path)
sys.path.append(project_root)

from yolo_seg import YoloSeg, YoloSegConfig


def main():
    parser = argparse.ArgumentParser(description="Ultralytics YOLO Segmentation Evaluation")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, default=0.7)
    parser.add_argument("--json-save-path", default="results_seg.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    coco = COCO(args.annotation)
    image_ids = coco.getImgIds()
    if args.limit > 0:
        image_ids = image_ids[:args.limit]

    coco_cat_ids = [cat["id"] for cat in coco.dataset["categories"]]

    model = YoloSeg(YoloSegConfig(model_path=args.model_path))
    results = []
    start = time.time()
    for image_id in image_ids:
        info = coco.loadImgs([image_id])[0]
        img = cv2.imread(os.path.join(args.image_dir, info["file_name"]))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        boxes, scores, cls_ids, masks = model(img, score_thres=args.conf_thres, nms_thres=args.nms_thres)
        for box, score, cls_id, mask in zip(boxes, scores, cls_ids, masks):
            x1, y1, x2, y2 = [float(v) for v in box]
            ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
            ix2, iy2 = min(int(x2), img_w), min(int(y2), img_h)
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if mask is not None and getattr(mask, 'size', 0) > 0 and ix2 > ix1 and iy2 > iy1:
                mh, mw = mask.shape[:2]
                th, tw = iy2 - iy1, ix2 - ix1
                if mh != th or mw != tw:
                    mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                full_mask[iy1:iy2, ix1:ix2] = (mask > 0).astype(np.uint8)
            encoded = mask_utils.encode(np.asfortranarray(full_mask))
            encoded["counts"] = encoded["counts"].decode("utf-8")
            results.append({
                "image_id": image_id,
                "category_id": int(coco_cat_ids[int(cls_id)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "segmentation": encoded,
            })

    with open(args.json_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(args.json_save_path)
    for iou_type in ["bbox", "segm"]:
        coco_eval = COCOeval(coco, coco_dt, iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")


if __name__ == "__main__":
    main()

