"""Offline input traversal, warmup, action dumps and complete/failed reports."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
from samples._shared.assets import sha256_file, verify_asset_file
from samples._shared.runtime_meta import metadata_evidence
from samples.robotics.himloco.runtime.python.input_io import (
    discover_inputs,
    load_observation,
)
from samples.robotics.himloco.runtime.python.model_runner import RuntimeModelRunner
from samples.robotics.himloco.runtime.python.policy import HimLocoTask


def _utc():
    return datetime.now(timezone.utc).isoformat()


def execute(args, selection):
    from samples._shared.platforms import require_execution_target

    require_execution_target(selection.target)
    model_digest = verify_asset_file(selection.asset, selection.model_path)
    records, manifest = discover_inputs(args.input_path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError("Output directory must be new")
    report_path = args.report or args.output_dir / "report.json"
    if (
        report_path.exists()
        or report_path.is_symlink()
        or report_path == args.output_dir
    ):
        raise ValueError("Report path must be new and different from output directory")
    # Only reserve our report; never replace an existing file even under a race.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as report_file:
        report = {
            "schema_version": "1.0",
            "status": "running",
            "started_utc": _utc(),
            "target": selection.target,
            "model": str(selection.model_path),
            "asset_id": selection.asset.reference,
            "model_sha256": model_digest,
            "input_manifest": manifest,
            "output_directory": str(args.output_dir),
            "sample_count": 0,
            "warmup_runs": args.warmup,
            "warmup_completed": 0,
            "records": [],
            "current_source_index": None,
            "scheduling": {"priority": args.priority, "bpu_cores": args.bpu_cores},
            "timing_scope": "synchronous bound runner including adapter validation/copy; excludes task pre/post, file I/O and warmup; not device-only latency",
            "environment": {
                "machine": platform.machine(),
                "python": sys.version.splitlines()[0],
                "numpy": np.__version__,
                "board_os_version": (
                    Path("/etc/version")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                    if Path("/etc/version").is_file()
                    else "unreported"
                ),
            },
        }

        def persist():
            report_file.seek(0)
            json.dump(report, report_file, indent=2, allow_nan=False)
            report_file.write("\n")
            report_file.truncate()
            report_file.flush()

        persist()
        try:
            runner = RuntimeModelRunner(selection)
            binding = runner.load()
            runner.set_scheduling_params(
                priority=args.priority, bpu_cores=args.bpu_cores
            )
            report["runtime"] = metadata_evidence(binding.metadata)
            sdk_module = sys.modules.get("hbm_runtime")
            report["runtime_module_source"] = str(
                getattr(sdk_module, "__file__", type(runner.runtime).__module__)
            )
            task = HimLocoTask(runner)
            first, _ = load_observation(records[0])
            for _ in range(args.warmup):
                task.predict(first)
                report["warmup_completed"] += 1
            latencies = []
            for record in records:
                report["current_source_index"] = record.source_index
                persist()
                values, digest = load_observation(record)
                result = task.predict(values)
                destination = args.output_dir / f"{record.source_index:06d}.bin"
                if destination == report_path:
                    raise ValueError("Report path conflicts with action dump")
                with destination.open("xb") as handle:
                    handle.write(result.actions.astype("<f4").tobytes())
                latencies.append(result.latency_ms)
                report["records"].append(
                    {
                        "source_index": record.source_index,
                        "input_file": str(record.path),
                        "input_sha256": digest,
                        "output_file": str(destination),
                        "output_sha256": sha256_file(destination),
                        "latency_ms": result.latency_ms,
                    }
                )
                report["sample_count"] = len(report["records"])
            if sha256_file(selection.model_path) != model_digest:
                raise ValueError("Model changed during execution")
            if (
                manifest is not None
                and sha256_file(Path(manifest["path"])) != manifest["sha256"]
            ):
                raise ValueError("Manifest changed during execution")
            values = np.asarray(latencies, dtype=np.float64)
            report["latency_ms"] = {
                "minimum": float(values.min()),
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(values.max()),
            }
            report["current_source_index"] = None
            report["status"] = "completed"
        except Exception as error:
            report["status"] = "failed"
            report["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            report["finished_utc"] = _utc()
            persist()
    return report
