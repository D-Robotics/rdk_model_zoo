#!/usr/bin/env python3
"""Compare fused HIMLoco TorchScript and ONNX on a held-out native rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from metrics import (
    load_rollout,
    numerical_metrics,
    select_indices,
    sha256,
    write_report,
)


INPUT_NAME = "obs_history"
OUTPUT_NAME = "actions"


def main() -> None:
    """Run held-out samples through both floating-point model formats."""

    parser = argparse.ArgumentParser(
        description="Compare HIMLoco policy.pt and fused ONNX on held-out rollout data."
    )
    parser.add_argument("--jit", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--data", required=True, type=Path, help="Native held-out rollout .pt."
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--observation-key", default=INPUT_NAME)
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Zero evaluates all samples."
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    args = parser.parse_args()

    jit_path = args.jit.resolve()
    onnx_path = args.onnx.resolve()
    data_path = args.data.resolve()
    report_path = args.report.resolve()
    for path in (jit_path, onnx_path, data_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in [-1,1]")
    if args.max_abs < 0.0:
        raise ValueError("--max-abs must be non-negative")

    observations, _ = load_rollout(data_path, args.observation_key)
    indices = select_indices(int(observations.shape[0]), args.num_samples, args.seed)
    jit_model = torch.jit.load(str(jit_path), map_location="cpu").eval()
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    if [value.name for value in onnx_model.graph.input] != [INPUT_NAME]:
        raise ValueError("ONNX must expose only the obs_history input")
    if [value.name for value in onnx_model.graph.output] != [OUTPUT_NAME]:
        raise ValueError("ONNX must expose only the actions output")
    evaluator = ReferenceEvaluator(onnx_model)

    reference_outputs = []
    candidate_outputs = []
    with torch.inference_mode():
        for source_index in indices.tolist():
            sample = observations[source_index : source_index + 1]
            reference = jit_model(sample)
            if not isinstance(reference, torch.Tensor):
                raise TypeError("TorchScript forward must return one Tensor")
            reference_array = reference.detach().cpu().numpy().astype(np.float32)
            candidate_array = evaluator.run(
                [OUTPUT_NAME], {INPUT_NAME: sample.numpy()}
            )[0].astype(np.float32)
            reference_outputs.append(reference_array[0])
            candidate_outputs.append(candidate_array[0])

    metrics = numerical_metrics(
        np.stack(reference_outputs), np.stack(candidate_outputs)
    )
    passed = bool(
        metrics["minimum_sample_cosine_similarity"] >= args.min_cosine
        and metrics["action_max_abs"] <= args.max_abs
    )
    report = {
        "schema_version": "1.0",
        "comparison": "fused TorchScript versus fused floating-point ONNX",
        "jit": str(jit_path),
        "jit_sha256": sha256(jit_path),
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "observation_key": args.observation_key,
        "selection": {
            "seed": args.seed,
            "sample_count": int(indices.size),
            "source_indices": indices.tolist(),
        },
        "thresholds": {
            "minimum_sample_cosine_similarity": args.min_cosine,
            "action_max_abs": args.max_abs,
        },
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__},
        "metrics": metrics,
        "passed": passed,
    }
    write_report(report_path, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
