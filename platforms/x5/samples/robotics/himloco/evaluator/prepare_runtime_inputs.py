#!/usr/bin/env python3
"""Prepare held-out HIMLoco rollout observations for RDK X5 Runtime transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import INPUT_WIDTH, load_rollout, select_indices, sha256

INPUT_BYTES = INPUT_WIDTH * np.dtype(np.float32).itemsize


def main() -> None:
    """Write source-indexed raw inputs without changing the native rollout.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(
        description="Prepare source-indexed X5 input dumps from a held-out rollout .pt."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--observation-key", default="obs_history")
    parser.add_argument(
        "--num-samples", type=int, default=0, help="Zero selects the complete rollout."
    )
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_path}")

    observations, _ = load_rollout(data_path, args.observation_key)
    indices = select_indices(int(observations.shape[0]), args.num_samples, args.seed)
    tensor_dir = output_path / "obs_history"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for source_index in indices.tolist():
        sample = observations[source_index].numpy().astype(np.float32, copy=False)
        sample_path = tensor_dir / f"{source_index:06d}.bin"
        sample.tofile(sample_path)
        if sample_path.stat().st_size != INPUT_BYTES:
            raise RuntimeError(f"unexpected input dump size: {sample_path}")
        records.append(
            {
                "source_index": source_index,
                "file": sample_path.relative_to(output_path).as_posix(),
                "sha256": sha256(sample_path),
                "bytes": INPUT_BYTES,
            }
        )

    manifest = {
        "schema_version": "1.0",
        "purpose": "held-out RDK X5 Runtime inputs",
        "source": str(data_path),
        "source_sha256": sha256(data_path),
        "source_format": "native PyTorch .pt; source file is not modified",
        "observation_key": args.observation_key,
        "input_contract": {
            "name": "obs_history",
            "shape": [1, INPUT_WIDTH],
            "dtype": "float32",
            "bytes_per_file": INPUT_BYTES,
        },
        "selection": {
            "source_sample_count": int(observations.shape[0]),
            "sample_count": len(records),
            "seed": args.seed,
            "source_indices": indices.tolist(),
        },
        "records": records,
    }
    manifest_path = output_path / "runtime-input-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"manifest": str(manifest_path), "samples": len(records)}, indent=2)
    )


if __name__ == "__main__":
    main()
