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

"""COCO keypoint evaluation for the Ultralytics YOLO pose models.

The metric definition is the standard COCO `keypoints` average precision
computed by `pycocotools.COCOeval`. Keypoints are emitted as the flat
`[x, y, v] * 17` list COCO expects; `v` is 1 when the model reports a positive
keypoint score and 0 otherwise, so an unconfident keypoint is excluded from
the metric rather than being counted at the origin.
"""

import argparse
import json
import os
import sys
import time

import cv2

_EVALUATOR_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_DIR = os.path.abspath(os.path.join(_EVALUATOR_DIR, os.pardir,
                                            "runtime", "python"))
for _path in (_EVALUATOR_DIR, _RUNTIME_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from eval_common import (  # noqa: E402  (path is set up above)
    add_platform_arguments,
    add_threshold_arguments,
    list_images,
    report_empty_predictions,
    resolve_platform_argument,
)
from yolo_pose import YoloPose, YoloPoseConfig  # noqa: E402


def flatten_keypoints(kpts_xy, kpts_score) -> list:
    """Flatten per-keypoint coordinates and scores into COCO's `[x, y, v]` form.

    Args:
        kpts_xy: Array of shape `(K, 2)` with keypoint coordinates.
        kpts_score: Array of shape `(K,)` with keypoint confidence scores.

    Returns:
        A flat list of `3 * K` floats.
    """
    flat = []
    for (x, y), score in zip(kpts_xy, kpts_score):
        value = score.item() if hasattr(score, "item") else float(score)
        flat.extend([float(x), float(y), 1 if value > 0 else 0])
    return flat


def build_parser() -> argparse.ArgumentParser:
    """Build the pose evaluator parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO pose evaluation on COCO "
                    "(metric: COCO keypoints AP).")
    parser.add_argument("--model-path", required=True,
                        help="Compiled .bin (X5) or .hbm (S) pose model.")
    parser.add_argument("--image-dir", required=True,
                        help="Directory holding the evaluation images.")
    parser.add_argument("--annotation", default=None,
                        help="COCO person-keypoints annotation JSON. When "
                             "omitted, predictions are written to "
                             "--json-save-path and no metric is computed.")
    parser.add_argument("--json-save-path", default="results_pose.json",
                        help="Where to write the COCO keypoint results.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N images; 0 means all.")
    parser.add_argument("--category-id", type=int, default=1,
                        help="COCO category id of the person class. Defaults "
                             "to 1, the standard COCO person id.")
    add_platform_arguments(parser)
    add_threshold_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Run the pose evaluation.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    platform = resolve_platform_argument(args)

    images = list_images(args.image_dir, args.annotation, args.limit)

    common = dict(model_path=args.model_path, platform=platform,
                  input_shape=args.input_shape)
    if args.conf_thres is not None:
        common["score_thres"] = args.conf_thres
    if args.nms_thres is not None:
        common["nms_thres"] = args.nms_thres
    model = YoloPose(YoloPoseConfig(**common))

    results = []
    start = time.time()
    for image_id, file_name in images:
        img = cv2.imread(os.path.join(args.image_dir, file_name))
        if img is None:
            raise FileNotFoundError(file_name)
        boxes, scores, _cls_ids, kpts_xy, kpts_score = model(
            img, score_thres=args.conf_thres, nms_thres=args.nms_thres)
        for box, score, kxy, kscore in zip(boxes, scores, kpts_xy, kpts_score):
            x1, y1, x2, y2 = (float(value) for value in box)
            results.append({
                "image_id": image_id,
                "category_id": args.category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "keypoints": flatten_keypoints(kxy, kscore),
            })

    with open(args.json_save_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)

    if args.annotation is None:
        print(f"wrote {len(results)} prediction(s) to {args.json_save_path}; "
              f"no --annotation was given, so no metric was computed.")
        return 0

    if not results:
        report_empty_predictions("pose", args.json_save_path)
        return 0

    coco = COCO(args.annotation)
    coco_dt = coco.loadRes(args.json_save_path)
    coco_eval = COCOeval(coco, coco_dt, "keypoints")
    coco_eval.params.imgIds = [image_id for image_id, _ in images]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
