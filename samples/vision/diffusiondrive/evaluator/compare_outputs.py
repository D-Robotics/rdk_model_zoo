# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Strict decoded-versus-float comparison; no dataset or board execution claims."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import sha256_file
from samples.vision.diffusiondrive.runtime.python.data_io import load_npz
from samples.vision.diffusiondrive.runtime.python.model_binding import OUTPUT_SHAPES


def cosine(lhs, rhs):
    if (
        lhs.shape != rhs.shape
        or not np.isfinite(lhs).all()
        or not np.isfinite(rhs).all()
    ):
        raise ValueError("Cosine requires identical shapes and finite arrays")
    a = lhs.astype(np.float64).reshape(-1)
    b = rhs.astype(np.float64).reshape(-1)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return None
    return float(np.clip(np.dot(a, b) / norm, -1, 1))


def class_distribution(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): float(c / labels.size) for v, c in zip(values, counts)}


def semantic_iou(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Label shapes differ")
    result = {}
    for c in range(7):
        a = reference == c
        b = candidate == c
        union = np.logical_or(a, b).sum()
        if union:
            result[c] = float(np.logical_and(a, b).sum() / union)
    return result


def _validate(values, shapes, dtypes):
    if set(values) != set(shapes):
        raise ValueError("Archive tensor names differ from schema")
    for name, shape in shapes.items():
        v = np.asarray(values[name])
        if (
            v.shape != shape
            or v.dtype != np.dtype(dtypes[name])
            or not np.isfinite(v).all()
        ):
            raise ValueError(
                f"Invalid tensor {name}: expected finite {dtypes[name]} {shape}"
            )


def compare(reference, candidate):
    _validate(reference, OUTPUT_SHAPES, {n: "float32" for n in OUTPUT_SHAPES})
    mapping = {
        "trajectory": "trajectory",
        "agent_states": "agent_states",
        "agent_labels": "agent_scores",
        "bev_semantic_map": "bev_logits",
    }
    shapes = {mapping[n]: s for n, s in OUTPUT_SHAPES.items()}
    shapes.update(agent_mask=(1, 30), bev_labels=(1, 128, 256))
    types = {n: "float32" for n in mapping.values()}
    types.update(agent_mask="bool", bev_labels="uint8")
    _validate(candidate, shapes, types)
    if np.any(candidate["agent_scores"] < 0) or np.any(candidate["agent_scores"] > 1):
        raise ValueError("Agent probabilities outside [0,1]")
    labels = candidate["bev_labels"]
    if np.any(labels > 6) or not np.array_equal(
        labels, np.argmax(candidate["bev_logits"], axis=1)
    ):
        raise ValueError(
            "BEV labels must be valid classes and match decoded-logit argmax"
        )
    metrics = {}
    for source, dest in mapping.items():
        a = reference[source]
        b = candidate[dest]
        if source == "agent_labels":
            a = 1 / (1 + np.exp(-np.clip(a, -60, 60)))
        metrics[source] = {
            "candidate_key": dest,
            "cosine": cosine(a, b),
            "mae": float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64)))),
            "max_abs_error": float(
                np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
            ),
        }
    ref_labels = np.argmax(reference["bev_semantic_map"], axis=1)
    iou = semantic_iou(ref_labels, labels)
    return {
        "schema_version": "1.0",
        "status": "descriptive; no acceptance threshold",
        "dataset_accuracy": False,
        "cosine_zero_norm": "null: undefined if either vector has zero norm",
        "tensors": metrics,
        "bev": {
            "pixel_agreement": float(np.mean(ref_labels == labels)),
            "reference_distribution": class_distribution(ref_labels),
            "candidate_distribution": class_distribution(labels),
            "class_iou": iou,
            "mean_iou": float(np.mean(list(iou.values()))),
            "averaging": "macro over classes present in either prediction, including background",
        },
        "scope": "one float-model reference versus decoded candidate, not ground truth or NAVSIM PDM Score",
    }


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-npz", required=True, type=Path)
    p.add_argument(
        "--board-npz", "--candidate-npz", dest="board_npz", required=True, type=Path
    )
    p.add_argument(
        "--output", type=Path, help="Optional new JSON path; default prints to stdout"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        output = args.output.expanduser().resolve() if args.output else None
        if output and output.exists():
            raise FileExistsError(f"Output report must be new: {output}")
        reference = args.reference_npz.expanduser().resolve()
        candidate = args.board_npz.expanduser().resolve()
        report = compare(load_npz(reference), load_npz(candidate))
        report.update(
            reference=str(reference),
            candidate=str(candidate),
            reference_sha256=sha256_file(reference),
            candidate_sha256=sha256_file(candidate),
        )
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text)
        print(text, end="")
        return 0
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
