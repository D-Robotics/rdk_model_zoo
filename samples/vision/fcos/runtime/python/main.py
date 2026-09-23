# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""SDK-free FCOS entrypoint and board execution boundary."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fcos.runtime.python.model_binding import (  # noqa: E402
    BindingError,
    SUPPORTED_TARGETS,
    list_available_assets,
    resolve_selection,
)

SAMPLE_DIR = ROOT / "samples" / "vision" / "fcos"
DEFAULT_IMAGE = SAMPLE_DIR / "test_data" / "bus.jpg"
DEFAULT_LABELS = ROOT / "datasets" / "coco" / "coco_classes.names"
DEFAULT_RESULT = SAMPLE_DIR / "test_data" / "result.jpg"


def build_parser() -> argparse.ArgumentParser:
    """Build the complete SDK-free FCOS CLI parser."""
    parser = argparse.ArgumentParser(description="FCOS object detection on RDK X5.")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", default=None, help="Exact manifest asset reference.")
    parser.add_argument("--variant", choices=("efficientnetb0", "efficientnetb2", "efficientnetb3"), default=None, help="Variant; omitted selects B0 unless --asset-id selects another exact row.")
    parser.add_argument("--model-path", default=None, help="External model path; requires exact --asset-id.")
    parser.add_argument("--test-img", default=str(DEFAULT_IMAGE), help="BGR image path.")
    parser.add_argument("--label-file", default=str(DEFAULT_LABELS), help="COCO label file (optional for numeric output).")
    parser.add_argument("--img-save-path", default=str(DEFAULT_RESULT), help="Annotated image output path.")
    parser.add_argument("--resize-type", type=int, choices=(0, 1), default=None)
    parser.add_argument("--classes-num", type=int, default=80)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.6)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List manifest assets without SDK access.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve one exact selection without SDK or board access.")
    return parser


def main(argv=None) -> int:
    """Run list, dry-run, or board inference and return a shell status."""
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            for record in list_available_assets(None if args.target == "auto" else args.target):
                print(record.asset_id)
            print("3 manifest assets; no model loaded.")
            return 0
        selection = resolve_selection(args.target, asset_id=args.asset_id, variant=args.variant, model_path=args.model_path)
        if args.classes_num != 80:
            raise ValueError("FCOS source contract has exactly 80 classes; --classes-num must remain 80.")
        if args.dry_run:
            print(json.dumps({
                "target": selection.target,
                "asset_id": selection.asset_id,
                "variant": selection.variant,
                "model_path": str(selection.model_path),
                "input": {"shape": [1, 3, selection.contract.input_height, selection.contract.input_width], "dtype": "NV12", "layout": "packed"},
                "outputs": {"classification_heads": 5, "box_heads": 5, "center_heads": 5, "strides": list(selection.contract.strides)},
                "no_sdk_loaded": True,
            }, indent=2))
            return 0
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        runner_module = importlib.import_module("samples.vision.fcos.runtime.python.model_runner")
        task_module = importlib.import_module("samples.vision.fcos.runtime.python.fcos")
        runner = runner_module.RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"input image not found or unreadable: {args.test_img}")
        task = task_module.FCOSTask(runner, binding, conf_thres=args.conf_thres, iou_thres=args.iou_thres, resize_type=args.resize_type)
        result = task.predict(np.asarray(image))
        labels = _load_labels(Path(args.label_file).expanduser())
        _save_result(Path(args.img_save_path).expanduser(), image, result, labels)
        print(json.dumps({"asset_id": selection.asset_id, "boxes": result.boxes.tolist(), "scores": result.scores.tolist(), "class_ids": result.class_ids.tolist(), "result_path": str(Path(args.img_save_path).expanduser())}))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _load_labels(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _save_result(path: Path, image: np.ndarray, result, labels: tuple[str, ...]) -> None:
    canvas = image.copy()
    for box, score, class_id in zip(result.boxes, result.scores, result.class_ids):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = labels[int(class_id)] if int(class_id) < len(labels) else str(int(class_id))
        cv2.putText(canvas, f"{label} {float(score):.3f}", (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write result image: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
