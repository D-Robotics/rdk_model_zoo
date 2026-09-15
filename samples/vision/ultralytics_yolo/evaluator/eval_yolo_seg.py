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

"""COCO instance segmentation evaluation.

Both the `bbox` and `segm` COCO metrics are reported, which is what the
published segmentation benchmarks were produced with. Per-instance masks are
pasted into a full-resolution binary mask before RLE encoding, so a prediction
whose box falls partly outside the image is clipped rather than dropped. An
empty or degenerate mask still produces a valid (all-zero) RLE entry.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

_EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_DIR = os.path.abspath(os.path.join(_EVALUATOR_DIR, os.pardir,
                                            "runtime", "python"))
for _path in (_EVALUATOR_DIR, _RUNTIME_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_common import (  # noqa: E402  (path is set up above)
    add_platform_arguments,
    add_threshold_arguments,
    category_ids,
    list_images,
    report_empty_predictions,
    resolve_platform_argument,
)
from yolo_seg import YoloSeg, YoloSegConfig  # noqa: E402


def encode_instance_mask(mask, box, img_w: int, img_h: int) -> dict:
    """Encode one instance mask as a COCO RLE dictionary.

    Args:
        mask: Predicted binary mask for the instance, or `None`/empty when the
            wrapper produced no mask for this detection.
        box: The instance box as `(x1, y1, x2, y2)`.
        img_w: Width of the original image.
        img_h: Height of the original image.

    Returns:
        A COCO RLE dictionary with a UTF-8 `counts` string, ready to be
        written to the results JSON.
    """
    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    x1, y1, x2, y2 = (float(value) for value in box)
    ix1, iy1 = max(int(x1), 0), max(int(y1), 0)
    ix2, iy2 = min(int(x2), img_w), min(int(y2), img_h)
    if mask is not None and getattr(mask, "size", 0) > 0 and ix2 > ix1 and iy2 > iy1:
        resized = mask
        mask_h, mask_w = mask.shape[:2]
        target_h, target_w = iy2 - iy1, ix2 - ix1
        if mask_h != target_h or mask_w != target_w:
            resized = cv2.resize(mask, (target_w, target_h),
                                 interpolation=cv2.INTER_NEAREST)
        full_mask[iy1:iy2, ix1:ix2] = (resized > 0).astype(np.uint8)
    from pycocotools import mask as mask_utils
    encoded = mask_utils.encode(np.asfortranarray(full_mask))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def build_parser() -> argparse.ArgumentParser:
    """Build the segmentation evaluator parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO instance segmentation evaluation on "
                    "COCO (metrics: COCO bbox AP and segm AP).")
    parser.add_argument("--model-path", required=True,
                        help="Compiled .bin (X5) or .hbm (S) segmentation model.")
    parser.add_argument("--image-dir", required=True,
                        help="Directory holding the evaluation images.")
    parser.add_argument("--annotation", default=None,
                        help="COCO annotation JSON of the evaluation split. "
                             "When omitted, predictions are written to "
                             "--json-save-path and no metric is computed.")
    parser.add_argument("--json-save-path", default="results_seg.json",
                        help="Where to write the COCO segmentation results.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N images; 0 means all.")
    add_platform_arguments(parser)
    add_threshold_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Run the segmentation evaluation.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    platform = resolve_platform_argument(args)

    images = list_images(args.image_dir, args.annotation, args.limit)
    coco_cat_ids = category_ids(args.annotation)

    common = dict(model_path=args.model_path, platform=platform,
                  input_shape=args.input_shape)
    if args.conf_thres is not None:
        common["score_thres"] = args.conf_thres
    if args.nms_thres is not None:
        common["nms_thres"] = args.nms_thres
    model = YoloSeg(YoloSegConfig(**common))

    results = []
    start = time.time()
    for image_id, file_name in images:
        img = cv2.imread(os.path.join(args.image_dir, file_name))
        if img is None:
            raise FileNotFoundError(file_name)
        img_h, img_w = img.shape[:2]
        boxes, scores, cls_ids, masks = model(
            img, score_thres=args.conf_thres, nms_thres=args.nms_thres)
        for box, score, cls_id, mask in zip(boxes, scores, cls_ids, masks):
            x1, y1, x2, y2 = (float(value) for value in box)
            results.append({
                "image_id": image_id,
                "category_id": int(coco_cat_ids[int(cls_id)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "segmentation": encode_instance_mask(mask, box, img_w, img_h),
            })

    with open(args.json_save_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)

    if args.annotation is None:
        print(f"wrote {len(results)} prediction(s) to {args.json_save_path}; "
              f"no --annotation was given, so no metric was computed.")
        return 0

    if not results:
        report_empty_predictions("seg", args.json_save_path)
        return 0

    coco = COCO(args.annotation)
    coco_dt = coco.loadRes(args.json_save_path)
    for iou_type in ("bbox", "segm"):
        coco_eval = COCOeval(coco, coco_dt, iou_type)
        coco_eval.params.imgIds = [image_id for image_id, _ in images]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
