# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Evaluate YOLOE PF through an explicit board or float ONNX backend and category map."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", required=True, choices=("onnx", "board"))
    p.add_argument("--target", required=True, choices=("x5", "s100", "s100p"))
    p.add_argument("--variant", required=True)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--annotation", required=True, type=Path)
    p.add_argument("--category-map", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--predictions-only", action="store_true")
    p.add_argument("--score-thres", type=float, default=0.25)
    p.add_argument("--nms-thres", type=float, default=None)
    p.add_argument("--resize-type", type=int, choices=(0, 1), default=1)
    p.add_argument("--no-morph", action="store_true")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--multi-label", action="store_true")
    p.add_argument(
        "--threads",
        type=int,
        default=2,
        help="ONNX CPU threads; no board scheduler change.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        from samples.vision.yoloe.evaluator.dataset import (
            load_dataset,
            load_category_map,
        )
        from samples.vision.yoloe.evaluator.backends import create_predictor, NAMES
        from samples.vision.yoloe.evaluator.engine import evaluate_dataset
        from samples.vision.yoloe.runtime.python.yoloe import Config
        from samples.vision.yoloe.runtime.python.model_binding import resolve_selection
        from samples.vision.yoloe.runtime.python.pipeline_io import validate_config

        selection = resolve_selection(args.target, variant=args.variant)
        cfg = Config(
            args.score_thres,
            args.nms_thres,
            args.resize_type,
            args.target != "x5" and args.variant.startswith("11") and not args.no_morph,
            args.max_det,
            not args.multi_label,
        )
        validate_config(selection, cfg)
        if args.output_dir.expanduser().exists():
            raise FileExistsError("Use a new evaluation output directory.")
        dataset = load_dataset(args.annotation, args.image_dir, args.limit)
        mapping = load_category_map(
            args.category_map, dataset.categories, NAMES.read_text().splitlines()
        )
        # RLE encoding is required even for prediction-only runs; fail before loading a model.
        from pycocotools import mask as _mask_utils

        predictor = create_predictor(
            args.backend,
            args.target,
            args.variant,
            args.model_path,
            args.model_sha256,
            cfg,
            args.threads,
        )
        result = evaluate_dataset(
            dataset,
            mapping,
            predictor,
            args.output_dir,
            predictions_only=args.predictions_only,
        )
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
