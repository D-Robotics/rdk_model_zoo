# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Dataset execution with input snapshots, per-image evidence and honest partial failures."""

from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import platform
from importlib.metadata import version, PackageNotFoundError
import cv2
import numpy as np
from utils.py_utils.assets import sha256_file
from samples.vision.yoloe.evaluator.results import (
    serialize_predictions,
    score_predictions,
)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def evaluate_dataset(
    dataset, mapping, predictor, output_dir, *, predictions_only=False
):
    """Fail on unreadable/mismatched images; no filtering of difficult images or categories."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    try:
        coco_version = version("pycocotools")
    except PackageNotFoundError:
        coco_version = None
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": predictor.identity,
        "annotation_input_sha256": dataset.annotation_sha256,
        "category_map_input_sha256": mapping.sha256,
        "image_root": str(dataset.image_root),
        "total_annotation_images": len(dataset.document["images"]),
        "selected_image_ids": [r["id"] for r in dataset.images],
        "category_ids": sorted(dataset.categories),
        "processed_images": 0,
        "predicted_instances": 0,
        "mapped_instances": 0,
        "unmapped_instances": 0,
        "metrics": None,
        "metric_scope": "not-run",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pycocotools": coco_version,
            "system": platform.platform(),
        },
        "timing_scope": "per-image wall time includes preprocessing, forward, postprocessing and COCO/RLE encoding; excludes image/JSON file I/O and is not BPU-only latency",
    }
    boxes = []
    masks = []
    try:
        write_json(root / "annotations.json", dataset.document)
        write_json(root / "category-map.json", mapping.document)
        report["annotation_snapshot_sha256"] = sha256_file(root / "annotations.json")
        report["category_map_snapshot_sha256"] = sha256_file(root / "category-map.json")
        with (root / "images.jsonl").open("w", encoding="utf-8") as stream:
            for row in dataset.images:
                path = (dataset.image_root / row["file_name"]).resolve()
                if not path.is_relative_to(dataset.image_root):
                    raise ValueError("Image path changed to escape image_root.")
                raw = path.read_bytes()
                image = (
                    cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if raw
                    else None
                )
                if image is None:
                    raise ValueError(f"Unreadable dataset image: {path}")
                if image.shape[:2] != (row["height"], row["width"]):
                    raise ValueError(f"Annotation/image dimensions differ: {path}")
                start = perf_counter()
                result = predictor.predict(image)
                b, s, dropped = serialize_predictions(
                    result, row["id"], image.shape[:2], mapping.ids
                )
                elapsed = perf_counter() - start
                boxes.extend(b)
                masks.extend(s)
                report["processed_images"] += 1
                report["predicted_instances"] += len(result.boxes)
                report["mapped_instances"] += len(b)
                report["unmapped_instances"] += dropped
                record = {
                    "image_id": row["id"],
                    "file_name": row["file_name"],
                    "sha256": sha256(raw).hexdigest(),
                    "height": row["height"],
                    "width": row["width"],
                    "predicted": len(result.boxes),
                    "mapped": len(b),
                    "unmapped": dropped,
                    "wall_seconds": elapsed,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
        write_json(root / "bbox-predictions.json", boxes)
        write_json(root / "segm-predictions.json", masks)
        report["prediction_sha256"] = {
            name: sha256_file(root / name)
            for name in ("bbox-predictions.json", "segm-predictions.json")
        }
        if predictions_only:
            report.update(status="predictions-only", metric_scope="not-requested")
        else:
            chosen = set(report["selected_image_ids"])
            if not any(
                a["image_id"] in chosen and not a.get("iscrowd", 0)
                for a in dataset.document.get("annotations", [])
            ):
                raise ValueError(
                    "Selected images have no non-crowd ground truth; use predictions-only instead of claiming accuracy."
                )
            with (root / "metrics.log").open("w") as log, redirect_stdout(log):
                report["metrics"] = score_predictions(
                    root / "annotations.json",
                    boxes,
                    masks,
                    report["selected_image_ids"],
                    report["category_ids"],
                )
            report.update(
                status="evaluated",
                metric_scope=(
                    "selected-images-only"
                    if len(dataset.images) < len(dataset.document["images"])
                    else "all-annotation-images"
                ),
            )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["processed_images"] != len(dataset.images):
            write_json(root / "bbox-predictions.partial.json", boxes)
            write_json(root / "segm-predictions.partial.json", masks)
            report["partial_predictions"] = [
                "bbox-predictions.partial.json",
                "segm-predictions.partial.json",
            ]
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "evaluation.json", report)
    return report
