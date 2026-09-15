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

"""ImageNet top-1 / top-5 evaluation for the Ultralytics YOLO classifiers.

The metric definition is the standard ImageNet single-crop top-1 and top-5
accuracy over the validation split: an image counts as top-1 when the highest
scoring class equals the ground truth, and as top-5 when the ground truth is
among the five highest scoring classes. The ground truth of each image is read
from one of two conventions:

    - `--val-txt FILE`, the ImageNet `val.txt` layout of `<path> <label>` per
      line. This is the S-series convention and the default when the file is
      given.
    - `--label-file FILE`, a file listing one synset id per line in class-index
      order, used to map the `n########` component of an X5-style file name.

Exactly one of the two must be supplied: with neither, the ground truth of an
image is unknowable and a guessed mapping would silently produce a wrong
accuracy, so the evaluator refuses to run.

The classification input resolution differs per platform. X5 publishes
`640x640` artifacts and the S series publishes `224x224` artifacts; both are
read from the model itself, so no resolution is assumed here.
"""

import argparse
import json
import os
import re
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
    report_empty_predictions,
    resolve_platform_argument,
)
from yolo_cls import YoloCls, YoloClsConfig  # noqa: E402

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_SYNSET_RE = re.compile(r"n\d+")


def load_val_txt(path: str, label_offset: int) -> dict:
    """Read an ImageNet `val.txt` ground-truth file.

    Args:
        path: Path of the `val.txt` file.
        label_offset: Value added to every label, for splits whose labels are
            1-based.

    Returns:
        A mapping of image base name to ground-truth class index.
    """
    truth = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 2:
                truth[os.path.basename(parts[0])] = int(parts[1]) + label_offset
    return truth


def load_synset_index(path: str) -> dict:
    """Read a synset-id-to-class-index file.

    Args:
        path: Path of a file listing one synset id (for example `n01440764`)
            per line, in class-index order.

    Returns:
        A mapping of synset id to class index.

    Raises:
        ValueError: If the file holds no usable synset id.
    """
    entries = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            token = line.strip().split()
            if token:
                entries.append(token[0])
    if not entries:
        raise ValueError(f"{path} lists no synset ids.")
    return {synset: index for index, synset in enumerate(entries)}


def image_ground_truth(name: str, synset_index: dict) -> int:
    """Derive the ground-truth class index of an X5-style image file name.

    Args:
        name: Image file name, expected to embed a synset id such as
            `n01440764`.
        synset_index: Mapping returned by `load_synset_index`.

    Returns:
        The class index.

    Raises:
        KeyError: If the file name embeds no known synset id.
    """
    match = _SYNSET_RE.search(name)
    if match is None:
        raise KeyError(name)
    return synset_index[match.group(0)]


def build_parser() -> argparse.ArgumentParser:
    """Build the classification evaluator parser.

    Returns:
        The configured `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Ultralytics YOLO classification evaluation on ImageNet "
                    "(metrics: top-1 and top-5 accuracy).")
    parser.add_argument("--model-path", required=True,
                        help="Compiled .bin (X5) or .hbm (S) classifier.")
    parser.add_argument("--image-dir", required=True,
                        help="Directory holding the validation images.")
    parser.add_argument("--json-save-path", default="results_cls.json",
                        help="Where to write the accuracy summary.")
    parser.add_argument("--val-txt", default=None,
                        help="ImageNet val.txt with '<path> <label>' lines.")
    parser.add_argument("--label-file", default=None,
                        help="File listing one synset id per line in class "
                             "index order; enables the n######## file-name "
                             "convention.")
    parser.add_argument("--label-offset", type=int, default=0,
                        help="Value added to every val.txt label.")
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of top classes returned per image.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N images; 0 means all.")
    parser.add_argument("--log-interval", type=int, default=1000,
                        help="Print a progress line every N images; 0 disables "
                             "progress output.")
    add_platform_arguments(parser)
    return parser


def main(argv=None) -> int:
    """Run the classification evaluation.

    Args:
        argv: Argument list, defaulting to `sys.argv[1:]`.

    Returns:
        Process exit status.

    Raises:
        SystemExit: If neither `--val-txt` nor `--label-file` is supplied.
    """
    args = build_parser().parse_args(argv)

    if not args.val_txt and not args.label_file:
        print("[Error] the ground truth of the validation images is unknown. "
              "Pass --val-txt (ImageNet val.txt) or --label-file (synset ids "
              "in class-index order). This evaluator does not guess a label "
              "mapping.", file=sys.stderr)
        return 2

    platform = resolve_platform_argument(args)
    val_truth = (load_val_txt(args.val_txt, args.label_offset)
                 if args.val_txt else {})
    synset_index = load_synset_index(args.label_file) if args.label_file else {}

    model = YoloCls(YoloClsConfig(model_path=args.model_path, platform=platform,
                                  input_shape=args.input_shape, topk=args.topk))

    names = sorted(name for name in os.listdir(args.image_dir)
                   if name.lower().endswith(_IMAGE_SUFFIXES))
    if args.limit > 0:
        names = names[:args.limit]

    top1 = 0
    top5 = 0
    total = 0
    start = time.time()
    for name in names:
        truth = val_truth.get(name)
        if truth is None and synset_index:
            try:
                truth = image_ground_truth(name, synset_index)
            except KeyError:
                truth = None
        if truth is None:
            continue
        img = cv2.imread(os.path.join(args.image_dir, name))
        if img is None:
            continue
        predictions = model(img, topk=args.topk)
        predicted_ids = [entry[0] for entry in predictions]
        total += 1
        if predicted_ids and predicted_ids[0] == truth:
            top1 += 1
        if truth in predicted_ids[:5]:
            top5 += 1
        if args.log_interval > 0 and total % args.log_interval == 0:
            print({
                "progress": total,
                "top1": top1 / total if total else 0.0,
                "top5": top5 / total if total else 0.0,
                "elapsed_sec": time.time() - start,
            }, flush=True)

    if total == 0:
        report_empty_predictions("cls", args.json_save_path)
        summary = {"total": 0, "top1": 0.0, "top5": 0.0,
                   "elapsed_sec": time.time() - start}
    else:
        summary = {
            "total": total,
            "top1": top1 / total,
            "top5": top5 / total,
            "elapsed_sec": time.time() - start,
        }
    with open(args.json_save_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
