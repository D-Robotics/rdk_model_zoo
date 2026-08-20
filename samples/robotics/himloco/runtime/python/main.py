#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run HIMLoco rollout input dumps through an RDK X5 BPU model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from himloco import INPUT_WIDTH, OUTPUT_WIDTH, HimLoco

INPUT_BYTES = INPUT_WIDTH * np.dtype(np.float32).itemsize


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(path: Path) -> list[tuple[int, Path]]:
    """Find raw input dumps and parse source rollout indexes from their stems."""

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob("*.bin"))
    else:
        raise FileNotFoundError(path)
    if not files:
        raise ValueError(f"no .bin input dumps found in {path}")

    records = []
    seen = set()
    for input_file in files:
        try:
            source_index = int(input_file.stem)
        except ValueError as error:
            raise ValueError(
                f"input dump stem must be a rollout source index: {input_file.name}"
            ) from error
        if source_index < 0:
            raise ValueError(f"source index must be non-negative: {source_index}")
        if source_index in seen:
            raise ValueError(f"duplicate source index: {source_index}")
        if input_file.stat().st_size != INPUT_BYTES:
            raise ValueError(
                f"{input_file} must contain exactly {INPUT_BYTES} float32 bytes"
            )
        records.append((source_index, input_file))
        seen.add(source_index)
    return sorted(records, key=lambda item: item[0])


def validate_input_manifest(
    input_path: Path, input_records: list[tuple[int, Path]]
) -> dict[str, str] | None:
    """Validate a colocated Runtime input manifest when one is available."""

    input_directory = input_path if input_path.is_dir() else input_path.parent
    manifest_path = input_directory.parent / "runtime-input-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("input_contract", {})
    if (
        contract.get("name") != "obs_history"
        or contract.get("shape") != [1, INPUT_WIDTH]
        or contract.get("dtype") != "float32"
        or contract.get("bytes_per_file") != INPUT_BYTES
    ):
        raise ValueError(f"input contract mismatch in {manifest_path}")

    manifest_records = {
        (int(record["source_index"]), Path(record["file"]).name): record
        for record in manifest.get("records", [])
    }
    for source_index, input_file in input_records:
        key = (source_index, input_file.name)
        if key not in manifest_records:
            raise ValueError(f"{input_file.name} is not recorded in {manifest_path}")
        if manifest_records[key].get("sha256") != sha256(input_file):
            raise ValueError(f"input hash mismatch for {input_file}")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "source": str(manifest.get("source", "unreported")),
        "source_sha256": str(manifest.get("source_sha256", "unreported")),
    }


def load_input(path: Path) -> np.ndarray:
    """Load one finite raw float32 policy input."""

    observation = np.fromfile(path, dtype=np.float32)
    if observation.size != INPUT_WIDTH:
        raise ValueError(f"unexpected input size in {path}: {observation.size}")
    if not np.isfinite(observation).all():
        raise ValueError(f"{path} contains NaN/Inf")
    return observation


def latency_summary(latencies: list[float]) -> dict[str, float]:
    """Summarize synchronous Runtime call latency in milliseconds."""

    values = np.asarray(latencies, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }


def environment_metadata() -> dict[str, str]:
    """Collect the board and Python environment recorded with each run."""

    version_path = Path("/etc/version")
    board_version = (
        version_path.read_text(encoding="utf-8", errors="replace").strip()
        if version_path.is_file()
        else "unreported"
    )
    return {
        "board_os_version": board_version,
        "machine": platform.machine(),
        "python": sys.version.splitlines()[0],
        "numpy": np.__version__,
    }


def main() -> None:
    """Validate the model contract, run input dumps, and write action evidence."""

    parser = argparse.ArgumentParser(
        description="Run fused HIMLoco policy input dumps on RDK X5."
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument(
        "--input-path",
        required=True,
        type=Path,
        help="One numerically named raw float32 .bin file or a directory of them.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--priority", type=int)
    parser.add_argument("--bpu-cores", nargs="+", type=int)
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if report_path.exists():
        raise FileExistsError(report_path)
    if report_path == output_dir:
        raise ValueError("--report and --output-dir must be different paths")

    input_records = discover_inputs(input_path)
    input_manifest = validate_input_manifest(input_path, input_records)
    model = HimLoco(model_path)
    model.set_scheduling_params(args.priority, args.bpu_cores)
    first_observation = load_input(input_records[0][1])
    model.warmup(first_observation, args.warmup)

    output_dir.mkdir(parents=True, exist_ok=True)
    latencies = []
    records = []
    for source_index, input_file in input_records:
        result = model.infer(load_input(input_file))
        output_file = output_dir / f"{source_index:06d}.bin"
        result.actions.tofile(output_file)
        if output_file.stat().st_size != OUTPUT_WIDTH * np.dtype(np.float32).itemsize:
            raise RuntimeError(f"unexpected action dump size: {output_file}")
        latencies.append(result.latency_ms)
        records.append(
            {
                "source_index": source_index,
                "input_file": str(input_file),
                "input_sha256": sha256(input_file),
                "output_file": str(output_file),
                "output_sha256": sha256(output_file),
                "latency_ms": result.latency_ms,
            }
        )

    report = {
        "schema_version": "1.0",
        "platform": "RDK X5",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "environment": environment_metadata(),
        "runtime": model.metadata(),
        "input_path": str(input_path),
        "input_manifest": input_manifest,
        "output_directory": str(output_dir),
        "sample_count": len(records),
        "warmup_runs": args.warmup,
        "scheduling": {
            "priority": args.priority,
            "bpu_cores": args.bpu_cores,
        },
        "timing_scope": (
            "synchronous HB_HBMRuntime.run call; input file reads, validation, "
            "and action dump writes excluded"
        ),
        "latency_ms": latency_summary(latencies),
        "records": records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
