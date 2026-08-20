#!/usr/bin/env python3
"""Convert native HIMLoco rollout tensors into Mapper float32 calibration files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


INPUT_WIDTH = 270
SAMPLE_BYTES = INPUT_WIDTH * np.dtype(np.float32).itemsize


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_observations(path: Path, key: str) -> torch.Tensor:
    """Load a native rollout tensor or a tensor stored under one dictionary key."""

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(artifact, torch.Tensor):
        observations = artifact
    elif isinstance(artifact, dict):
        if key not in artifact:
            raise KeyError(f"{key!r} is not present in {path}")
        observations = artifact[key]
    else:
        raise TypeError("rollout .pt must contain a Tensor or a dictionary of Tensors")
    if not isinstance(observations, torch.Tensor):
        raise TypeError(f"{key!r} must be a Tensor")
    if observations.ndim != 2 or observations.shape[1] != INPUT_WIDTH:
        raise ValueError(
            f"expected {key} shape [N, {INPUT_WIDTH}], got {tuple(observations.shape)}"
        )
    observations = observations.detach().cpu().to(torch.float32).contiguous()
    if observations.shape[0] == 0:
        raise ValueError("rollout tensor is empty")
    if not torch.isfinite(observations).all():
        invalid = int((~torch.isfinite(observations)).sum())
        raise ValueError(f"rollout contains {invalid} NaN/Inf values")
    return observations


def main() -> None:
    """Select deterministic rollout samples and write raw Mapper inputs."""

    parser = argparse.ArgumentParser(
        description="Prepare HIMLoco obs_history calibration data from a native .pt rollout."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Native rollout .pt file."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New calibration root."
    )
    parser.add_argument(
        "--tensor-key",
        default="obs_history",
        help="Dictionary key containing the [N,270] policy inputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples; use 0 to select the complete rollout.",
    )
    parser.add_argument("--seed", type=int, default=20260820, help="Sampling RNG seed.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_path}")

    observations = load_observations(input_path, args.tensor_key)
    available = int(observations.shape[0])
    requested = available if args.num_samples == 0 else args.num_samples
    if requested < 1 or requested > available:
        raise ValueError(f"--num-samples must be in [1, {available}] or 0")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(available, size=requested, replace=False))
    tensor_dir = output_path / "obs_history"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    records = []
    selected_values = []
    for ordinal, source_index in enumerate(indices.tolist()):
        sample = observations[source_index].numpy().astype(np.float32, copy=False)
        sample_path = tensor_dir / f"{ordinal:06d}.bin"
        sample.tofile(sample_path)
        if sample_path.stat().st_size != SAMPLE_BYTES:
            raise ValueError(f"unexpected raw tensor size: {sample_path}")
        selected_values.append(sample)
        records.append(
            {
                "ordinal": ordinal,
                "source_index": source_index,
                "file": sample_path.relative_to(output_path).as_posix(),
                "sha256": sha256(sample_path),
                "bytes": SAMPLE_BYTES,
                "minimum": float(sample.min()),
                "maximum": float(sample.max()),
                "mean": float(sample.mean()),
                "standard_deviation": float(sample.std()),
            }
        )

    selected = np.stack(selected_values).astype(np.float32)
    manifest = {
        "schema_version": "1.0",
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "source_tensor_key": args.tensor_key,
        "source_sample_count": available,
        "selection": {
            "method": "seeded random sampling without replacement, sorted by source index",
            "seed": args.seed,
            "sample_count": requested,
            "source_indices": indices.tolist(),
        },
        "input_contract": {
            "name": "obs_history",
            "model_shape": [1, INPUT_WIDTH],
            "file_shape": [INPUT_WIDTH],
            "dtype": "float32",
            "layout": "flat featuremap",
            "normalization": "none; values are captured at the policy input boundary",
            "bytes_per_file": SAMPLE_BYTES,
        },
        "aggregate_statistics": {
            "minimum": float(selected.min()),
            "maximum": float(selected.max()),
            "mean": float(selected.mean()),
            "standard_deviation": float(selected.std()),
            "non_finite_values": 0,
        },
        "records": records,
    }
    manifest_path = output_path / "calibration-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = f"""# HIMLoco Calibration Preprocess Report

- Source: `{input_path}`
- Source SHA256: `{manifest["source_sha256"]}`
- Tensor key: `{args.tensor_key}`
- Source samples: {available}
- Selected samples: {requested}
- Selection seed: {args.seed}
- Model input: `obs_history`, float32 `[1, 270]`
- Per-file payload: {SAMPLE_BYTES} bytes
- Normalization: none; the files preserve values captured at the policy input boundary
- NaN/Inf values: 0

The original rollout remains in its native `.pt` representation. The files under
`obs_history/` are deterministic raw float32 build inputs for OpenExplorer Mapper;
they are generated data and must not be committed to the repository.
"""
    (output_path / "preprocess-report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "samples": requested}, indent=2))


if __name__ == "__main__":
    main()
