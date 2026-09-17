#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Evaluate PaddleOCR JSON/JSONL records with deterministic IoU matching.

The evaluator intentionally operates on saved records and has no dependency
on an OCR runtime, model files, or a dataset downloader.  Matching is greedy in
ground-truth record order; for a ground-truth region, the highest-IoU unused
prediction over the threshold wins, with prediction order breaking ties.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class EvaluationError(ValueError):
    """Raised when an evaluation record does not satisfy the file contract."""


def evaluate_records(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate ordered image records and return JSON-compatible metrics."""

    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise EvaluationError("iou_threshold must be a finite number in [0, 1].")
    gt = _normalise_records(ground_truth, "ground truth")
    pred = _normalise_records(predictions, "prediction")
    gt_by_id = _index_records(gt, "ground truth")
    pred_by_id = _index_records(pred, "prediction")

    # Preserve GT order, then append prediction-only records in prediction
    # order.  This makes per-image output stable while still counting false
    # positives when the prediction file contains an extra image.
    order = [record["image"] for record in gt]
    order.extend(
        record["image"] for record in pred if record["image"] not in gt_by_id
    )

    per_image: list[dict[str, Any]] = []
    total_gt = total_pred = total_matched = exact_matches = 0
    similarity_sum = 0.0
    for image_id in order:
        gt_record = gt_by_id.get(image_id, {"image": image_id, "boxes": [], "texts": []})
        pred_record = pred_by_id.get(image_id, {"image": image_id, "boxes": [], "texts": []})
        matches = _match_image(gt_record, pred_record, iou_threshold)
        image_gt = len(gt_record["boxes"])
        image_pred = len(pred_record["boxes"])
        total_gt += image_gt
        total_pred += image_pred
        total_matched += len(matches)
        exact_matches += sum(1 for match in matches if match["exact"])
        similarity_sum += sum(match["normalized_similarity"] for match in matches)
        per_image.append(
            {
                "image": image_id,
                "ground_truth_regions": image_gt,
                "predicted_regions": image_pred,
                "matched_regions": len(matches),
                "matches": matches,
            }
        )

    precision = _ratio(total_matched, total_pred)
    recall = _ratio(total_matched, total_gt)
    f1 = _ratio(2 * precision * recall, precision + recall)
    recognition = {
        "matched_regions": total_matched,
        "exact_matches": exact_matches,
        "exact_rate": _ratio(exact_matches, total_matched),
        "normalized_similarity": (
            similarity_sum / total_matched if total_matched else 0.0
        ),
        "status": "measured" if total_matched else "not_run",
    }
    return {
        "iou_threshold": float(iou_threshold),
        "images": len(order),
        "ground_truth_regions": total_gt,
        "predicted_regions": total_pred,
        "matched_regions": total_matched,
        "unmatched_ground_truth": total_gt - total_matched,
        "unmatched_predictions": total_pred - total_matched,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "recognition": recognition,
        "per_image": per_image,
    }


def load_records(path: str | Path) -> list[Mapping[str, Any]]:
    """Load one JSON object, a JSON array, or JSONL records from ``path``."""

    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"could not read records file {source}: {exc}") from exc
    if not text.strip():
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"invalid JSON on line {line_number} of {source}: {exc.msg}"
                ) from exc
            records.append(_record_object(item, source, line_number))
        return records
    if isinstance(value, list):
        return [_record_object(item, source, index + 1) for index, item in enumerate(value)]
    return [_record_object(value, source, 1)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the file-based evaluator and write a JSON report."""

    parser = argparse.ArgumentParser(
        description="Evaluate PaddleOCR JSON/JSONL records with IoU matching."
    )
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum axis-aligned polygon IoU for a match (default: 0.5).",
    )
    parser.add_argument("--output", type=Path, help="Optional output JSON report path.")
    args = parser.parse_args(argv)
    try:
        report = evaluate_records(
            load_records(args.ground_truth),
            load_records(args.predictions),
            iou_threshold=args.iou_threshold,
        )
    except EvaluationError as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


def _record_object(value: Any, source: Path, line_number: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(
            f"record {line_number} of {source} must be a JSON object."
        )
    return value


def _normalise_records(
    records: Sequence[Mapping[str, Any]], label: str
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        image_id = record.get("image", record.get("image_id", record.get("id", index)))
        if isinstance(image_id, (dict, list)):
            raise EvaluationError(f"{label} record {index} has an invalid image id.")
        boxes_value = record.get("boxes", [])
        texts_value = record.get("texts", [])
        if not isinstance(boxes_value, Sequence) or isinstance(boxes_value, (str, bytes)):
            raise EvaluationError(f"{label} record {index} boxes must be a list.")
        if not isinstance(texts_value, Sequence) or isinstance(texts_value, (str, bytes)):
            raise EvaluationError(f"{label} record {index} texts must be a list.")
        boxes = tuple(_box(value, label, index, box_index) for box_index, value in enumerate(boxes_value))
        texts = tuple(str(value) for value in texts_value)
        if len(boxes) != len(texts):
            raise EvaluationError(
                f"{label} record {index} has {len(boxes)} boxes but {len(texts)} texts."
            )
        normalised.append({"image": image_id, "boxes": boxes, "texts": texts})
    return normalised


def _index_records(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[Any, Mapping[str, Any]]:
    result: dict[Any, Mapping[str, Any]] = {}
    for record in records:
        image_id = record["image"]
        try:
            duplicate = image_id in result
        except TypeError as exc:
            raise EvaluationError(f"{label} image ids must be hashable.") from exc
        if duplicate:
            raise EvaluationError(f"{label} contains duplicate image id {image_id!r}.")
        result[image_id] = record
    return result


def _box(value: Any, label: str, record_index: int, box_index: int) -> tuple[float, float, float, float]:
    points = value
    # OpenCV-style contours are sometimes serialized as [[[x, y], ...]].
    if (
        isinstance(points, Sequence)
        and len(points) == 1
        and isinstance(points[0], Sequence)
        and points[0]
        and isinstance(points[0][0], Sequence)
    ):
        points = points[0]
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise EvaluationError(f"{label} record {record_index} box {box_index} is not a polygon.")
    coordinates: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            raise EvaluationError(f"{label} record {record_index} box {box_index} has an invalid point.")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"{label} record {record_index} box {box_index} has non-numeric coordinates.") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise EvaluationError(f"{label} record {record_index} box {box_index} has non-finite coordinates.")
        coordinates.append((x, y))
    if len(coordinates) < 2:
        raise EvaluationError(f"{label} record {record_index} box {box_index} needs at least two points.")
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    return min(xs), min(ys), max(xs), max(ys)


def _match_image(
    gt_record: Mapping[str, Any],
    pred_record: Mapping[str, Any],
    threshold: float,
) -> list[dict[str, Any]]:
    used: set[int] = set()
    matches: list[dict[str, Any]] = []
    gt_boxes = gt_record["boxes"]
    pred_boxes = pred_record["boxes"]
    for gt_index, gt_box in enumerate(gt_boxes):
        best_index: int | None = None
        best_iou = threshold
        for pred_index, pred_box in enumerate(pred_boxes):
            if pred_index in used:
                continue
            overlap = _iou(gt_box, pred_box)
            # At threshold zero, keep disjoint boxes from becoming matches;
            # a match still needs a meaningful positive intersection.
            if overlap > 0 and overlap >= threshold and (
                best_index is None or overlap > best_iou
            ):
                best_index = pred_index
                best_iou = overlap
        if best_index is None:
            continue
        used.add(best_index)
        gt_text = str(gt_record["texts"][gt_index])
        pred_text = str(pred_record["texts"][best_index])
        matches.append(
            {
                "ground_truth_index": gt_index,
                "prediction_index": best_index,
                "iou": best_iou,
                "ground_truth_text": gt_text,
                "prediction_text": pred_text,
                "exact": gt_text == pred_text,
                "normalized_similarity": _text_similarity(gt_text, pred_text),
            }
        )
    return matches


def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _text_similarity(first: str, second: str) -> float:
    if first == second:
        return 1.0
    width = max(len(first), len(second), 1)
    previous = list(range(len(second) + 1))
    for row, left in enumerate(first, 1):
        current = [row]
        for column, right in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return 1.0 - previous[-1] / width


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


__all__ = ["EvaluationError", "evaluate_records", "load_records", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
