# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Binary keyword metrics; undefined denominators remain null, never perfect."""

import math


def evaluate(records, threshold=0.5):
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite in [0,1]")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty prediction list")
    seen = set()
    counts = dict(tp=0, tn=0, fp=0, fn=0)
    for row in records:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or row["id"] in seen
        ):
            raise ValueError("Prediction IDs must be unique nonempty strings")
        seen.add(row["id"])
        label = row.get("label")
        score = row.get("score")
        if (
            type(label) is not int
            or label not in (0, 1)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Expected label 0/1 and finite probability score")
        predicted = score >= threshold
        counts[
            ("t" if predicted == bool(label) else "f") + ("p" if predicted else "n")
        ] += 1
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    ratio = lambda a, b: a / b if b else None
    return {
        "count": len(records),
        "threshold": threshold,
        "threshold_rule": "score >= threshold",
        "confusion": counts,
        "accuracy": (tp + tn) / len(records),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_accept_rate": ratio(fp, fp + tn),
        "false_reject_rate": ratio(fn, fn + tp),
    }
