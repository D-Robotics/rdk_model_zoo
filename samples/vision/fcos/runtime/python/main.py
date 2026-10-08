# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the FCOS command-line sample.

Parse options, construct the runner and the detection task, call
``predict``, and present the result. Option declarations, listing/dry-run,
label reading, and drawing live in ``cli.py``; the detection stages and
per-call geometry live in ``fcos.py``; the manifest and tensor contract in
``model_binding.py``. Help, listing, and dry-run stay model-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fcos.runtime.python.cli import (  # noqa: F401 - build_parser re-exported for the contract checker
    build_parser,
    dry_run,
    list_models,
    load_labels,
    save_result,
)
from samples.vision.fcos.runtime.python.model_binding import BindingError, resolve_selection


def main(argv=None) -> int:
    """Run list, dry-run, or board inference and return a shell status.

    Args:
        argv: Optional argument sequence excluding the program name; None
            reads sys.argv through argparse.

    Returns:
        int: 0 for a successful run; 2 for a reported selection, option,
        IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return list_models(args.target)
        selection = resolve_selection(args.target, asset_id=args.asset_id, variant=args.variant, model_path=args.model_path)
        if args.classes_num != 80:
            raise ValueError("FCOS source contract has exactly 80 classes; --classes-num must remain 80.")
        if args.dry_run:
            return dry_run(selection, args)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        import cv2

        from samples.vision.fcos.runtime.python.fcos import FCOSTask

        task = FCOSTask(selection, conf_thres=args.conf_thres, iou_thres=args.iou_thres, resize_type=args.resize_type)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"input image not found or unreadable: {args.test_img}")
        result = task.predict(image)
        labels = load_labels(Path(args.label_file).expanduser())
        save_result(Path(args.img_save_path).expanduser(), image, result, labels)
        print(json.dumps({"asset_id": selection.asset_id, "boxes": result.boxes.tolist(), "scores": result.scores.tolist(), "class_ids": result.class_ids.tolist(), "result_path": str(Path(args.img_save_path).expanduser())}))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
