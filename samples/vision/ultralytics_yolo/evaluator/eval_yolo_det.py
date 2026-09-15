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

"""COCO bbox evaluation for the Ultralytics YOLO detection models.

The metric definition is the standard COCO `bbox` average precision computed
by `pycocotools.COCOeval`, which is what the published detection benchmarks
were produced with. Category IDs follow standard COCO class order, including annotation subsets.

Only S-series YOLOv10 uses the NMS-free path. X5 retains its original DFL
decoder with NMS. Filename recognition selects the family.
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
    evaluation_types,
    evaluation_options,
    add_threshold_arguments,
    category_ids,
    list_images,
    report_empty_predictions,
    resolve_platform_argument,
)
from yolo_assets import family_from_filename  # noqa: E402
from yolo_detect import YoloDetect, YoloDetectConfig  # noqa: E402
from yolo_v10detect import YoloV10Detect, YoloV10DetectConfig  # noqa: E402


def build_model(args, platform):
    """Instantiate the detection wrapper matching the model file name.

    Args:
        args: Parsed evaluator arguments.
        platform: Resolved platform profile.

    Returns:
        A `(model, is_nms_free)` tuple. `is_nms_free` is True for YOLOv10,
        which is evaluated without an NMS threshold.
    """
    Model,Config=evaluation_types(args,platform,'detect')
    common=dict(model_path=args.model_path,platform=platform,input_shape=args.input_shape,**evaluation_options(args,'detect'))
    if args.conf_thres is not None:common['score_thres']=args.conf_thres
    nms_free=Model.__name__=='YoloV10Detect'
    if not nms_free and args.nms_thres is not None:common['nms_thres']=args.nms_thres
    return Model(Config(**common)),nms_free


def build_parser() -> argparse.ArgumentParser:
    """Build the detection evaluator parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO detection evaluation on COCO "
                    "(metric: COCO bbox AP).")
    parser.add_argument("--model-path", required=True,
                        help="Compiled .bin (X5) or .hbm (S) detection model.")
    parser.add_argument("--image-dir", required=True,
                        help="Directory holding the evaluation images.")
    parser.add_argument("--annotation", default=None,
                        help="COCO annotation JSON of the evaluation split. "
                             "When omitted, predictions are written to "
                             "--json-save-path and no metric is computed.")
    parser.add_argument("--json-save-path", default="results_det.json",
                        help="Where to write the COCO detection results.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N images; 0 means all.")
    add_platform_arguments(parser)
    add_threshold_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Run the detection evaluation.

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
    coco_cat_ids = category_ids(args.annotation)

    model, is_nms_free = build_model(args, platform)
    results = []
    start = time.time()

    for image_id, file_name in images:
        img = cv2.imread(os.path.join(args.image_dir, file_name))
        if img is None:
            raise FileNotFoundError(file_name)
        if is_nms_free:
            boxes, scores, cls_ids = model(img, score_thres=args.conf_thres)
        else:
            boxes, scores, cls_ids = model(img, score_thres=args.conf_thres,
                                           nms_thres=args.nms_thres)
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = (float(value) for value in box)
            results.append({
                "image_id": image_id,
                "category_id": int(coco_cat_ids[int(cls_id)]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

    with open(args.json_save_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle)

    if args.annotation is None:
        print(f"wrote {len(results)} prediction(s) to {args.json_save_path}; "
              f"no --annotation was given, so no metric was computed.")
        return 0

    if not results:
        report_empty_predictions("detect", args.json_save_path)
        return 0

    coco = COCO(args.annotation)
    coco_dt = coco.loadRes(args.json_save_path)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.params.imgIds = [image_id for image_id, _ in images]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print(f"elapsed: {time.time() - start:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
