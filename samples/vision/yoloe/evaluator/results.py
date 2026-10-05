# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Strict result serialization and COCO scoring; no inferred PF-to-dataset labels."""

from copy import deepcopy
import numpy as np


def serialize_predictions(result, image_id, image_shape, mapping):
    """Encode full/ROI masks without resampling; count explicitly unmapped detections."""
    from pycocotools import mask as mask_utils

    height, width = image_shape
    boxes = np.asarray(result.boxes)
    scores = np.asarray(result.scores)
    classes = np.asarray(result.class_ids)
    n = len(boxes)
    if (
        boxes.shape != (n, 4)
        or scores.shape != (n,)
        or classes.shape != (n,)
        or len(result.masks) != n
    ):
        raise ValueError(
            "Prediction arrays and masks must have matching detection counts."
        )
    if (
        not np.issubdtype(classes.dtype, np.integer)
        or np.any(classes < 0)
        or np.any(classes >= 4585)
    ):
        raise ValueError("Prediction class IDs must be integers in the PF vocabulary.")
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or np.any(scores > 1)
    ):
        raise ValueError(
            "Predictions require finite boxes and probability scores in [0,1]."
        )
    if result.mask_layout not in ("full", "roi"):
        raise ValueError("Unknown mask layout.")
    bbox_rows = []
    mask_rows = []
    dropped = 0
    for box, score, pf, mask in zip(boxes, scores, classes, result.masks):
        x1, y1, x2, y2 = map(float, box)
        if not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height:
            raise ValueError("Box lies outside its original image or is reversed.")
        mask = np.asarray(mask)
        if (
            mask.dtype not in (np.dtype("bool"), np.dtype("uint8"))
            or not np.isin(mask, [0, 1]).all()
        ):
            raise ValueError("Masks must be binary bool/uint8 arrays.")
        if result.mask_layout == "full":
            if mask.shape != (height, width):
                raise ValueError("Full mask shape differs from image.")
            full = np.asarray(mask, dtype=np.uint8)
        else:
            ix1, iy1, ix2, iy2 = map(int, (x1, y1, x2, y2))
            if mask.shape != (iy2 - iy1, ix2 - ix1):
                raise ValueError(
                    "ROI mask shape differs from clipped integer box; no automatic resizing."
                )
            full = np.zeros((height, width), np.uint8)
            full[iy1:iy2, ix1:ix2] = mask
        if int(pf) not in mapping:
            dropped += 1
            continue
        common = {
            "image_id": int(image_id),
            "category_id": mapping[int(pf)],
            "score": float(score),
        }
        bbox_rows.append(common | {"bbox": [x1, y1, x2 - x1, y2 - y1]})
        rle = mask_utils.encode(np.asfortranarray(full))
        rle["counts"] = rle["counts"].decode("ascii")
        # Keep segmentation results free of bbox: COCO loadRes must derive mask
        # area from the RLE rather than substituting rectangle area.
        mask_rows.append(common | {"segmentation": rle})
    return bbox_rows, mask_rows, dropped


def score_predictions(annotation, bboxes, masks, image_ids, category_ids):
    """Compute bbox/segm AP/AR including all-negative predictions; retain -1 undefined stats."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation))
    # The public loader accepts annotation documents without optional COCO
    # metadata, but pycocotools loadRes reads dataset['info'] unconditionally
    # on some versions (2.0.10). COCO() re-parses the file into a private
    # in-memory dict, so normalizing here cannot touch the annotation file or
    # any caller-held document; a provided 'info' is preserved as-is.
    ground_truth.dataset.setdefault("info", {})
    metrics = {}
    names = (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    )
    for kind, rows in [("bbox", bboxes), ("segm", masks)]:
        if rows:
            detections = ground_truth.loadRes(deepcopy(rows))
        else:
            detections = COCO()
            detections.dataset = {
                "images": deepcopy(ground_truth.dataset["images"]),
                "categories": deepcopy(ground_truth.dataset["categories"]),
                "annotations": [],
            }
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, kind)
        evaluator.params.imgIds = list(image_ids)
        evaluator.params.catIds = list(category_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[kind] = {
            name: float(value) for name, value in zip(names, evaluator.stats)
        }
        metrics[kind]["parameters"] = {
            "iou_thresholds": evaluator.params.iouThrs.tolist(),
            "recall_threshold_count": len(evaluator.params.recThrs),
            "max_detections": list(evaluator.params.maxDets),
            "area_labels": list(evaluator.params.areaRngLbl),
            "image_ids": [int(i) for i in evaluator.params.imgIds],
            "category_ids": [int(i) for i in evaluator.params.catIds],
        }
    return metrics
