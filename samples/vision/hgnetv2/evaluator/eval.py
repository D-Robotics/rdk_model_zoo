#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Evaluate HGNetV2 using recursive images and relative-path CSV labels.

The source's successful-inference denominator and direct-resize default are
preserved. Failed/unlabelled images are counted explicitly; an empty run does
not manufacture zero accuracy. Model loading remains behind the CLI boundary.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_ground_truth_csv(csv_path: str | Path) -> dict[str, int]:
    """Read relative image paths and zero-based ImageNet labels.

    Args:
        csv_path: UTF-8 CSV with optional image:file/category header.

    Returns:
        Normalized relative path to class ID; backslashes become slashes.

    Raises:
        ValueError: A row has an invalid label, missing path or conflicting ID.
    """
    labels: dict[str, int] = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        for number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                continue
            if (number == 1 and len(row) >= 2 and
                    row[0].strip().lower() in {"image:file", "image_path", "image"} and
                    row[1].strip().lower() == "category"):
                continue
            if len(row) < 2 or not row[0].strip():
                raise ValueError(f"CSV row {number}: expected relative path and category")
            path = row[0].strip().replace("\\", "/")
            try:
                label = int(row[1].strip())
            except ValueError as exc:
                raise ValueError(f"CSV row {number}: invalid category {row[1]!r}") from exc
            if not 0 <= label < 1000:
                raise ValueError(f"CSV row {number}: category must be in [0, 999]")
            if path in labels and labels[path] != label:
                raise ValueError(f"CSV row {number}: conflicting category for {path}")
            labels[path] = label
    return labels


def collect_images_with_relative_paths(image_root: str | Path) -> list[tuple[str, str]]:
    """Return sorted (relative path, absolute path) pairs for JPEG/PNG files."""
    root = Path(image_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"image directory not found: {root}")
    return sorted(
        (p.relative_to(root).as_posix(), str(p))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def evaluate_images(
    images: Sequence[tuple[str, str]],
    ground_truth: Mapping[str, int],
    predict: Callable,
    *,
    top_k: int = 5,
) -> dict:
    """Evaluate a callable task and report accuracy on successful images.

    The timer includes image reads and the complete prediction pipeline. It
    excludes model loading and directory/CSV scans. FPS is end-to-end loop
    throughput, not isolated BPU latency. Errors retain the affected path.
    """
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k must be between 1 and 1000")
    import cv2

    matched = successful = top1 = topk = 0
    errors: list[dict[str, str]] = []
    start = perf_counter()
    for relative, absolute in images:
        label = ground_truth.get(relative)
        if label is None:
            continue
        matched += 1
        try:
            image = cv2.imread(absolute, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("image is unreadable")
            result = predict(image)
            ids = result.class_ids.tolist()
            if len(ids) != top_k:
                raise ValueError(f"expected {top_k} predictions, got {len(ids)}")
        except Exception as exc:
            errors.append({"image": relative, "error": str(exc)})
            continue
        successful += 1
        top1 += int(ids[0] == label)
        topk += int(label in ids)
    elapsed = perf_counter() - start
    status = ("no-results" if successful == 0 else
              "partial" if errors or matched != len(images) else "complete")
    summary = {
        "status": status,
        "total_images_scanned": len(images),
        "matched_to_gt": matched,
        "unmatched_images": len(images) - matched,
        "successful_inferences": successful,
        "failed_images": len(errors),
        "errors": errors,
        "top1_acc": top1 / successful if successful else None,
        "topk_acc": topk / successful if successful else None,
        "top_k": top_k,
        "elapsed_seconds": elapsed,
        "fps": successful / elapsed if elapsed > 0 else 0.0,
        "accuracy_denominator": "successful_inferences",
        "timing_scope": "image reads + preprocessing + inference + postprocessing; no warmup",
    }
    if top_k == 5:
        summary["top5_acc"] = summary["topk_acc"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create an SDK-free evaluator parser, retaining source option aliases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("auto", "x5"), default="auto", help="Execution target (default: auto).")
    parser.add_argument("--variant", choices=("b0", "b1", "b2", "b3", "b4"), default=None, help="Published variant (default: b0).")
    parser.add_argument("--asset-id", help="Exact manifest reference for an external model path.")
    parser.add_argument("--model-path", help="Existing model path; requires --asset-id.")
    parser.add_argument("--image-path", required=True, help="Root of validation images (recursive).")
    parser.add_argument("--val-csv", required=True, help="CSV: relative image path, zero-based category.")
    parser.add_argument("--label-file", default="", help="Optional class-name file (default: no names).")
    parser.add_argument("--json-save-path", default="hgnetv2_cls_results.json", help="Results JSON path.")
    parser.add_argument("--limit", type=int, default=0, help="First N sorted images, before GT matching (0: all).")
    parser.add_argument("--top-k", "--topk", dest="top_k", type=int, default=5, help="Accuracy K, 1–1000 (default: 5).")
    parser.add_argument("--resize-type", type=int, choices=(0, 1), default=0, help="0 direct / 1 letterbox (source evaluator default: 0).")
    parser.add_argument("--priority", type=int, default=0, help="Scheduling priority (0–255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU cores (default: 0).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run evaluation, save coverage-aware JSON and return 2 on failures."""
    args = build_parser().parse_args(argv)
    try:
        if args.limit < 0 or not 1 <= args.top_k <= 1000:
            raise ValueError("limit must be nonnegative and top-k must be 1–1000")
        labels = load_ground_truth_csv(args.val_csv)
        images = collect_images_with_relative_paths(args.image_path)
        if args.limit:
            images = images[:args.limit]
        if not any(relative in labels for relative, _ in images):
            raise ValueError("no scanned image matches the CSV relative paths")
        from samples.vision.hgnetv2.runtime.python.model_binding import resolve_selection
        from samples.vision.hgnetv2.runtime.python.model_runner import RuntimeModelRunner
        from samples.vision.hgnetv2.runtime.python.classification import ClassificationTask
        from samples.vision.hgnetv2.runtime.python.labels import load_labels
        selection = resolve_selection(args.target, variant=args.variant,
                                      asset_id=args.asset_id, model_path=args.model_path)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = ClassificationTask(runner, binding, top_k=args.top_k,
                                  resize_type=args.resize_type,
                                  labels=load_labels(Path(args.label_file)) if args.label_file else None)
        summary = evaluate_images(images, labels, task.predict, top_k=args.top_k)
        summary.update({
            "date": datetime.now(timezone.utc).isoformat(),
            "model": str(selection.model_path), "asset_id": selection.asset_id,
            "target": selection.target, "image_root": str(Path(args.image_path).resolve()),
            "csv_file": str(Path(args.val_csv).resolve()),
            "config": {"resize_type": args.resize_type, "topk": args.top_k,
                       "limit": args.limit, "bpu_cores": args.bpu_cores, "priority": args.priority},
        })
        output = Path(args.json_save_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["successful_inferences"] and not summary["failed_images"] else 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
