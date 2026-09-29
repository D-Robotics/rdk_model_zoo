# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source depth metric formulas with explicit float64 pixel-pooled accumulation.

Preserves source lower-median alignment; numerical summation is not claimed
bit-identical to the original Torch float32 reductions. Invalid observations
are never silently advertised as perfect accuracy.
"""

import numpy as np

METRICS = ("delta1", "delta2", "delta3", "abs_rel", "rmse", "silog")


def lower_median(values):
    return np.partition(values, (len(values) - 1) // 2)[(len(values) - 1) // 2]


class DepthMetrics:
    def __init__(self, align, min_depth=0.001, max_depth=100.0):
        if align not in ("none", "median") or not 0 < min_depth < max_depth:
            raise ValueError("Invalid depth alignment/range")
        self.align, self.min_depth, self.max_depth = align, min_depth, max_depth
        self.totals = np.zeros(7, np.float64)
        self.count = 0
        self.scales = []

    def update(self, prediction, target):
        pred, gt = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
        if (
            pred.ndim != 2
            or pred.shape != gt.shape
            or not pred.size
            or not np.isfinite(pred).all()
        ):
            raise ValueError(
                "Expected matching nonempty depth planes and finite predictions"
            )
        valid = np.isfinite(gt) & (gt > self.min_depth) & (gt < self.max_depth)
        count = int(valid.sum())
        if not count:
            return {**{k: None for k in METRICS}, "valid_pixels": 0, "scale": None}
        p, g = pred[valid], gt[valid]
        scale = 1.0
        if self.align == "median":
            scale = float(lower_median(g) / lower_median(np.maximum(p, self.min_depth)))
            p = p * scale
        p = np.clip(p, self.min_depth, self.max_depth)
        threshold = np.maximum(p / g, g / p)
        log_diff = np.log(p) - np.log(g)
        totals = np.array(
            [
                (threshold < 1.25).sum(),
                (threshold < 1.25**2).sum(),
                (threshold < 1.25**3).sum(),
                (np.abs(p - g) / g).sum(),
                ((p - g) ** 2).sum(),
                (log_diff**2).sum(),
                log_diff.sum(),
            ],
            np.float64,
        )
        self.totals += totals
        self.count += count
        self.scales.append(scale)
        single = DepthMetrics(self.align, self.min_depth, self.max_depth)
        single.totals, single.count, single.scales = totals, count, [scale]
        return {**single.result(), "scale": scale}

    def result(self):
        if not self.count:
            return {**{k: None for k in METRICS}, "valid_pixels": 0}
        d1, d2, d3, rel, sq, log_sq, log_sum = self.totals / self.count
        return {
            "delta1": float(d1),
            "delta2": float(d2),
            "delta3": float(d3),
            "abs_rel": float(rel),
            "rmse": float(np.sqrt(sq)),
            "silog": float(np.sqrt(max(log_sq - log_sum**2, 0)) * 100),
            "valid_pixels": self.count,
            "median_scale_mean": float(np.mean(self.scales)),
            "median_scale_std": float(np.std(self.scales)),
        }


def fidelity_metrics(candidate, reference):
    if not candidate or len(candidate) != len(reference):
        raise ValueError("Fidelity requires equal nonempty sample sets")
    for a, b in zip(candidate, reference):
        if (
            np.shape(a) != np.shape(b)
            or not np.size(a)
            or not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError(
                "Fidelity requires matching finite arrays; values are not silently dropped"
            )
    a = np.concatenate([np.asarray(v, np.float64).ravel() for v in candidate])
    b = np.concatenate([np.asarray(v, np.float64).ravel() for v in reference])
    delta = a - b
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "max_abs": float(np.abs(delta).max()),
        "mean_relative_abs": float(
            (np.abs(delta) / np.maximum(np.abs(b), 1e-6)).mean()
        ),
        "cosine_similarity": float(np.dot(a, b) / denominator) if denominator else None,
        "finite_values": int(a.size),
    }
