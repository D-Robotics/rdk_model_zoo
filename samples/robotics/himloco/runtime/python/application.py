"""Offline input traversal, warmup, action dumps and complete/failed reports.

The workflow is split so the entry can show the model construction and the
per-observation ``predict`` loop while this module keeps the evidence
discipline: :func:`prepare` gates the target, collects input digests and
reserves the (new) report file, :func:`load_task` constructs the bound
policy task, :meth:`PreparedRun.record_sample` writes one action dump and
its evidence record, :func:`complete` re-verifies digests and summarizes
latencies, and :func:`execute` keeps the established single-call composition
of the same helpers.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
from utils.py_utils.assets import sha256_file, verify_asset_file
from utils.py_utils.runtime_meta import metadata_evidence
from samples.robotics.himloco.runtime.python.input_io import (
    discover_inputs,
    load_observation,
)
from samples.robotics.himloco.runtime.python.model_runner import RuntimeModelRunner
from samples.robotics.himloco.runtime.python.policy import HimLocoTask


def _utc():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PreparedRun:
    """Evidence-collected run: the reserved report file, records and digests.

    Holds everything the entry loop and :func:`execute` need between
    preparation and completion. ``persist`` rewrites the reserved report file
    in place; ``close`` stamps the finish time, persists once more and
    releases the file. Only ``close`` closes the report file.
    """

    args: object
    selection: object
    report: dict
    report_file: object
    report_path: Path
    output_dir: Path
    records: tuple
    manifest: object
    model_digest: str

    def persist(self):
        self.report_file.seek(0)
        json.dump(self.report, self.report_file, indent=2, allow_nan=False)
        self.report_file.write("\n")
        self.report_file.truncate()
        self.report_file.flush()

    def mark_failed(self, error):
        self.report["status"] = "failed"
        self.report["error"] = {"type": type(error).__name__, "message": str(error)}

    def close(self):
        """Stamp the finish time, persist once more and release the file.

        Like a file object's own ``close``, repeated calls are harmless.
        """

        if self.report_file.closed:
            return
        self.report["finished_utc"] = _utc()
        try:
            self.persist()
        finally:
            self.report_file.close()


def prepare(args, selection) -> PreparedRun:
    """Gate the target, collect input/model evidence and reserve the report.

    The established evidence order is preserved: board identity and asset
    digest first, input discovery next, then the new-directory and new-report
    checks; the report skeleton is persisted before any model is loaded. The
    report file is opened exclusively and stays open until
    :meth:`PreparedRun.close`.
    """
    from utils.py_utils.platforms import require_execution_target

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
    report_file = report_path.open("x", encoding="utf-8")
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
    run = PreparedRun(
        args,
        selection,
        report,
        report_file,
        report_path,
        args.output_dir,
        records,
        manifest,
        model_digest,
    )
    try:
        run.persist()
    except Exception:
        report_file.close()
        raise
    return run


def load_task(run: PreparedRun) -> HimLocoTask:
    """Construct the bound policy task and record the runtime evidence."""

    runner = RuntimeModelRunner(run.selection)
    binding = runner.load()
    runner.set_scheduling_params(
        priority=run.args.priority, bpu_cores=run.args.bpu_cores
    )
    run.report["runtime"] = metadata_evidence(binding.metadata)
    sdk_module = sys.modules.get("hbm_runtime")
    run.report["runtime_module_source"] = str(
        getattr(sdk_module, "__file__", type(runner.runtime).__module__)
    )
    return HimLocoTask(runner)


def record_sample(run: PreparedRun, record, result, digest) -> Path:
    """Write one source-indexed action dump and append its evidence record."""

    destination = run.output_dir / f"{record.source_index:06d}.bin"
    if destination == run.report_path:
        raise ValueError("Report path conflicts with action dump")
    with destination.open("xb") as handle:
        handle.write(result.actions.astype("<f4").tobytes())
    run.report["records"].append(
        {
            "source_index": record.source_index,
            "input_file": str(record.path),
            "input_sha256": digest,
            "output_file": str(destination),
            "output_sha256": sha256_file(destination),
            "latency_ms": result.latency_ms,
        }
    )
    run.report["sample_count"] = len(run.report["records"])
    return destination


def complete(run: PreparedRun, latencies) -> None:
    """Re-verify input digests, summarize latencies and mark completion."""

    if sha256_file(run.selection.model_path) != run.model_digest:
        raise ValueError("Model changed during execution")
    if (
        run.manifest is not None
        and sha256_file(Path(run.manifest["path"])) != run.manifest["sha256"]
    ):
        raise ValueError("Manifest changed during execution")
    values = np.asarray(latencies, dtype=np.float64)
    run.report["latency_ms"] = {
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(values.max()),
    }
    run.report["current_source_index"] = None
    run.report["status"] = "completed"


def execute(args, selection):
    """Compatibility composition: prepare, load, predict loop, complete."""
    run = prepare(args, selection)
    try:
        task = load_task(run)
        first, _ = load_observation(run.records[0])
        for _ in range(args.warmup):
            task.predict(first)
            run.report["warmup_completed"] += 1
        latencies = []
        for record in run.records:
            run.report["current_source_index"] = record.source_index
            run.persist()
            values, digest = load_observation(record)
            result = task.predict(values)
            record_sample(run, record, result, digest)
            latencies.append(result.latency_ms)
        complete(run, latencies)
    except Exception as error:
        run.mark_failed(error)
        raise
    finally:
        run.close()
    return run.report
