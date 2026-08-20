#!/usr/bin/env python3
"""Compare X5 HIMLoco action dumps with held-out reference actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metrics import load_rollout, numerical_metrics, sha256, write_report


OUTPUT_WIDTH = 12
OUTPUT_BYTES = OUTPUT_WIDTH * np.dtype(np.float32).itemsize


def load_candidate_dumps(
    directory: Path, sample_count: int
) -> tuple[np.ndarray, list[int], list[dict]]:
    """Load float32 action dumps whose numeric stems identify source samples."""

    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = sorted(directory.glob("*.bin"))
    if not files:
        raise ValueError(f"no .bin action dumps found in {directory}")

    outputs = []
    indices = []
    records = []
    seen = set()
    for path in files:
        try:
            source_index = int(path.stem)
        except ValueError as error:
            raise ValueError(
                f"action dump stem must be a source index: {path.name}"
            ) from error
        if source_index in seen:
            raise ValueError(f"duplicate source index: {source_index}")
        if source_index < 0 or source_index >= sample_count:
            raise IndexError(
                f"source index {source_index} is outside [0,{sample_count})"
            )
        if path.stat().st_size != OUTPUT_BYTES:
            raise ValueError(f"{path} must contain exactly {OUTPUT_BYTES} bytes")
        output = np.fromfile(path, dtype=np.float32)
        if not np.isfinite(output).all():
            raise ValueError(f"{path} contains NaN/Inf")
        outputs.append(output)
        indices.append(source_index)
        records.append(
            {"source_index": source_index, "file": path.name, "sha256": sha256(path)}
        )
        seen.add(source_index)
    return np.stack(outputs), indices, records


def main() -> None:
    """Evaluate candidate output dumps against JIT or captured rollout actions."""

    parser = argparse.ArgumentParser(
        description="Compare X5 float32 action dumps with HIMLoco held-out references."
    )
    parser.add_argument(
        "--data", required=True, type=Path, help="Native held-out rollout .pt."
    )
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--jit", type=Path, help="Reference policy.pt; otherwise use data actions."
    )
    parser.add_argument("--observation-key", default="obs_history")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="Optional maximum action-space absolute error gate.",
    )
    args = parser.parse_args()

    data_path = args.data.resolve()
    candidate_dir = args.candidate_dir.resolve()
    report_path = args.report.resolve()
    jit_path = args.jit.resolve() if args.jit else None
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if jit_path is not None and not jit_path.is_file():
        raise FileNotFoundError(jit_path)
    if not -1.0 <= args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in [-1,1]")
    if args.max_abs is not None and args.max_abs < 0.0:
        raise ValueError("--max-abs must be non-negative")

    observations, recorded_actions = load_rollout(
        data_path,
        args.observation_key,
        None if jit_path is not None else args.action_key,
    )
    candidates, indices, records = load_candidate_dumps(
        candidate_dir, int(observations.shape[0])
    )
    selected_observations = observations[indices]
    if jit_path is not None:
        jit_model = torch.jit.load(str(jit_path), map_location="cpu").eval()
        reference_batches = []
        with torch.inference_mode():
            for sample in selected_observations:
                output = jit_model(sample.unsqueeze(0))
                if not isinstance(output, torch.Tensor):
                    raise TypeError("TorchScript forward must return one Tensor")
                reference_batches.append(output.detach().cpu().numpy()[0])
        references = np.stack(reference_batches).astype(np.float32)
        reference_kind = "fused TorchScript"
    else:
        if recorded_actions is None:
            raise KeyError(f"{args.action_key!r} is required when --jit is omitted")
        references = recorded_actions[indices].numpy()
        reference_kind = f"recorded rollout key {args.action_key!r}"

    metrics = numerical_metrics(references, candidates)
    passed = bool(metrics["minimum_sample_cosine_similarity"] >= args.min_cosine)
    if args.max_abs is not None:
        passed = passed and bool(metrics["action_max_abs"] <= args.max_abs)
    report = {
        "schema_version": "1.0",
        "comparison": f"X5 action dumps versus {reference_kind}",
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "candidate_directory": str(candidate_dir),
        "reference_jit": str(jit_path) if jit_path else None,
        "reference_jit_sha256": sha256(jit_path) if jit_path else None,
        "observation_key": args.observation_key,
        "action_key": None if jit_path else args.action_key,
        "thresholds": {
            "minimum_sample_cosine_similarity": args.min_cosine,
            "action_max_abs": args.max_abs,
        },
        "records": records,
        "metrics": metrics,
        "passed": passed,
    }
    write_report(report_path, report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
