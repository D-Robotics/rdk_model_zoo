# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Evaluate saved raw/calibrated depth arrays; this tool never runs a board model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from samples._shared.assets import sha256_file
from samples.vision.yolo26_depth.runtime.python.geometry import make_context
from samples.vision.yolo26_depth.runtime.python.tensor_io import (
    restore_log_depth,
    resize_opencv,
)
from samples.vision.yolo26_depth.runtime.python.model_binding import LITE_CALIBRATION
from samples.vision.yolo26_depth.evaluator.metrics import (
    DepthMetrics,
    fidelity_metrics,
    METRICS,
)

CALIBRATION = {
    "n": (1.0, -0.19384765625),
    "s": (1.0, 1.71484375),
    "m": (1.0, 1.630859375),
    **LITE_CALIBRATION,
}
PROTOCOLS = ("deployment_letterbox", "deployment_scale_fill", "ultralytics_validator")


def resize_torch(depth, height, width):
    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))[None, None]
    return F.interpolate(
        tensor, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].numpy()


def decode_output(raw, record, protocol, boundary, variant, resize_backend="opencv"):
    if (
        variant not in CALIBRATION
        or protocol not in PROTOCOLS
        or boundary not in ("log", "raw")
    ):
        raise ValueError("Unknown variant/protocol/boundary")
    if (boundary == "raw") != (protocol == "deployment_scale_fill"):
        raise ValueError(
            "Output boundary conflicts with source protocol: raw requires scale-fill; log requires letterbox/validator"
        )
    value = np.asarray(raw)
    if (
        value.shape not in ((192, 192), (1, 192, 192, 1), (1, 1, 192, 192))
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            "Expected finite floating-point 192-square single-channel model output"
        )
    log = value.astype(np.float32).reshape(192, 192)
    profile = "lite" if boundary == "raw" else "nv12"
    if boundary == "raw":
        a, b = CALIBRATION[variant]
        log = np.clip(log, -4, 5) * a + b
    height, width = (
        (768, 768) if protocol == "ultralytics_validator" else record["original_hw"]
    )
    ctx = make_context(height, width, profile, variant)
    if resize_backend not in ("opencv", "torch"):
        raise ValueError("Unknown resize backend")
    return restore_log_depth(
        log, ctx, resize=resize_opencv if resize_backend == "opencv" else resize_torch
    )


def load_output(path, key):
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive or "indices" not in archive:
            raise ValueError(f"{path} requires indices and {key}")
        indices, values = archive["indices"], archive[key]
        if (
            indices.ndim != 1
            or not len(indices)
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                "Output indices must be nonempty unique nonnegative integers"
            )
        if values.ndim < 2 or len(values) != len(indices):
            raise ValueError("Output values and indices have different lengths")
        if values.dtype.kind != "f" or not np.isfinite(values).all():
            raise ValueError("Model outputs must be finite floating-point arrays")
        return indices.copy(), {
            int(index): values[position].copy()
            for position, index in enumerate(indices)
        }


def ground_truth(root, record, protocol):
    if not record.get("depth_m"):
        raise ValueError(f'Missing ground truth for record {record["index"]}')
    path = root / record["depth_m"]
    depth = np.load(path, allow_pickle=False).astype(np.float32)
    if depth.ndim != 2 or tuple(depth.shape) != tuple(record["original_hw"]):
        raise ValueError(f"Ground-truth geometry mismatch: {path}")
    if protocol == "ultralytics_validator":
        height, width = record["ultralytics_validator"]["stage1_hw"]
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (768, 768), interpolation=cv2.INTER_NEAREST)
    return depth


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-manifest", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--float-outputs", type=Path, required=True)
    p.add_argument("--quant-outputs", type=Path, required=True)
    p.add_argument("--candidate-name", required=True)
    p.add_argument("--protocol", choices=PROTOCOLS, required=True)
    p.add_argument("--boundary", choices=("log", "raw"), required=True)
    p.add_argument("--variant", choices=tuple(CALIBRATION), required=True)
    p.add_argument("--resize-backend", choices=("opencv", "torch"), default="opencv")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args(argv)
    if args.report.exists():
        raise FileExistsError(args.report)
    prepared = json.loads(args.prepared_manifest.read_text())
    if prepared.get("size") != 768 or args.protocol not in prepared.get(
        "protocols", {}
    ):
        raise ValueError(
            "Prepared manifest must declare the selected 768-square protocol"
        )
    records = {int(r["index"]): r for r in prepared["records"]}
    if len(records) != len(prepared["records"]):
        raise ValueError("Prepared record indices are not unique")
    suffix = "log" if args.boundary == "log" else "raw"
    fi, floats = load_output(args.float_outputs, "float_" + suffix)
    qi, quants = load_output(args.quant_outputs, "quant_" + suffix)
    if set(map(int, fi)) != set(map(int, qi)) or not set(map(int, qi)) <= set(records):
        raise ValueError(
            "Candidate/reference indices must match exactly and exist in prepared records"
        )
    selected = sorted(map(int, qi))
    acc = {
        f"{kind}_{align}": DepthMetrics(align)
        for kind in ("float", "quant")
        for align in ("none", "median")
    }
    per_sample, fd, qd = [], [], []
    for index in selected:
        record = records[index]
        gt = ground_truth(args.source_root, record, args.protocol)
        a = decode_output(
            floats[index],
            record,
            args.protocol,
            args.boundary,
            args.variant,
            args.resize_backend,
        )
        b = decode_output(
            quants[index],
            record,
            args.protocol,
            args.boundary,
            args.variant,
            args.resize_backend,
        )
        item = {
            "index": index,
            "sample": record["sample"],
            "sensor": record.get("sensor", "unknown"),
        }
        for kind, value in (("float", a), ("quant", b)):
            for align in ("none", "median"):
                item[f"{kind}_{align}"] = acc[f"{kind}_{align}"].update(value, gt)
        fd.append(a)
        qd.append(b)
        per_sample.append(item)
    fm = {a: acc[f"float_{a}"].result() for a in ("none", "median")}
    qm = {a: acc[f"quant_{a}"].result() for a in ("none", "median")}
    report = {
        "schema_version": "2.0",
        "candidate": args.candidate_name,
        "protocol": args.protocol,
        "boundary": args.boundary,
        "variant": args.variant,
        "resize_backend": args.resize_backend,
        "board": "not-run by this offline tool",
        "sample_count": len(selected),
        "indices": selected,
        "input_provenance": "Protocol and boundary are caller declarations; saved outputs are not independently proven to derive from the prepared tensors",
        "metric_contract": {
            "min_depth": 0.001,
            "max_depth": 100.0,
            "aggregation": "float64 pixel-pooled",
            "median_alignment": "per-image lower median(gt)/lower median(max(pred,0.001))",
            "prediction_clamp": [0.001, 100.0],
            "no_valid_pixels": "metrics null, not perfect scores",
        },
        "float_vs_gt": fm,
        "quant_vs_gt": qm,
        "accuracy_delta_quant_minus_float": {
            a: {
                k: (
                    qm[a][k] - fm[a][k]
                    if qm[a][k] is not None and fm[a][k] is not None
                    else None
                )
                for k in METRICS
            }
            for a in ("none", "median")
        },
        "quant_vs_float": {
            "log_depth" if args.boundary == "log" else "raw_logit": fidelity_metrics(
                [quants[i] for i in selected], [floats[i] for i in selected]
            ),
            "postprocessed_depth": fidelity_metrics(qd, fd),
        },
        "per_sample": per_sample,
    }
    for name in ("prepared_manifest", "float_outputs", "quant_outputs"):
        path = getattr(args, name)
        report[name], report[name + "_sha256"] = str(path), sha256_file(path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)
    args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
