# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface and offline evidence records for the HIMLoco sample.

Option declarations, argument validation, the published X5 asset identity
and selection, the source-indexed observation records, the model-free
listing/dry-run rendering and the evidence discipline live here:
:func:`prepare` gates the target, collects input digests and reserves the
(new) report file, :meth:`PreparedRun.record_sample` writes one action dump
and its evidence record, and :func:`complete` re-verifies digests and
summarizes latencies. ``main.py`` stays a thin, readable entry that
constructs the bound policy task visibly and drives the offline predict
loop. Nothing in this module imports NumPy at module level or loads a board
SDK, so host listing, help and dry-run stay light. The policy tensor binding
and task stages live in ``policy.py``.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

from utils.py_utils.assets import Asset, list_assets, sha256_file, verify_asset_file
from utils.py_utils.runtime_meta import metadata_evidence

SAMPLE_DIR = Path(__file__).resolve().parents[2]
ASSET_ID = "x5:himloco:himloco_go2_bayese_1x270.bin"


@dataclass(frozen=True)
class ModelSelection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def resolve_selection(target="auto", *, model_path=None, asset_id=None):
    """Resolve exact X5 asset; alternate paths require its explicit asset identity."""
    if target == "auto":
        from utils.py_utils.platforms import detect_target

        target = detect_target()
    if target != "x5":
        raise ValueError("HIMLoco has a published fused model only for x5")
    assets = list_assets("x5", "himloco")
    if len(assets) != 1 or assets[0].reference != ASSET_ID or assets[0].format != "bin":
        raise ValueError("Expected the single published HIMLoco BIN asset")
    asset = assets[0]
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Expected asset-id {asset.reference}")
    if model_path is not None and asset_id is None:
        raise ValueError(
            "An external model path requires the explicit matching asset-id"
        )
    path = (
        Path(model_path).expanduser()
        if model_path is not None
        else SAMPLE_DIR / "model/bayes-e" / asset.filename
    )
    if path.suffix != ".bin":
        raise ValueError("HIMLoco requires an X5 .bin artifact")
    return ModelSelection(target, asset, path, model_path is not None)


def validate_selection(selection):
    expected = resolve_selection(
        selection.target,
        model_path=selection.model_path if selection.explicit_model_path else None,
        asset_id=selection.asset.reference,
    )
    if selection != expected:
        raise ValueError("Selection differs from the declared publication and path")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def build_parser():
    """Build the SDK-free command line parser for the sample entrypoint."""

    parser = argparse.ArgumentParser(
        description="HIMLoco: published X5 model selection and source-indexed "
                    "offline inference."
    )
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--input-path", type=Path, default=SAMPLE_DIR / "test_data/obs_history"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/himloco"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--priority", type=int)
    parser.add_argument("--bpu-cores", type=int, nargs="+")
    return parser


def normalize_args(args) -> None:
    """Validate flags in place and expand every path argument.

    Mutually exclusive modes are handled by the parser; the numeric gates,
    path expansion and the ``--list-models`` target defaulting happen here,
    before any selection resolution, in the established order.
    """
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.priority is not None and not 0 <= args.priority <= 255:
        raise ValueError("priority must be in [0,255]")
    if args.bpu_cores is not None and any(c < 0 for c in args.bpu_cores):
        raise ValueError("BPU cores must be nonnegative")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.list_models and args.target == "auto":
        args.target = "x5"


def print_resolution(args, selection) -> int:
    """Print the model-free listing/dry-run report (no SDK, no downloads)."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "url": selection.asset.url,
                "sha256": selection.asset.sha256,
                "input_path": str(args.input_path),
                "sdk_loaded": False,
                "downloaded": False,
                "metadata_verified": False,
            },
            indent=2,
        )
    )
    return 0


@dataclass
class PreparedRun:
    """Evidence-collected run: the reserved report file, records and digests.

    Holds everything the entry loop needs between preparation and
    completion. ``persist`` rewrites the reserved report file in place;
    ``close`` stamps the finish time, persists once more and releases the
    file. Only ``close`` closes the report file.
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
    # Imported here so the model-free modes of this module stay NumPy-free.
    import numpy as np

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


def note_runtime(run: PreparedRun, task) -> None:
    """Record the bound SDK metadata of a constructed policy task.

    Args:
        run: PreparedRun whose report receives the runtime evidence.
        task: HimLocoTask constructed through ``HimLocoTask.from_model``;
            its ``metadata`` and ``runtime_module_source`` properties supply
            the evidence.

    Returns:
        None.
    """

    run.report["runtime"] = metadata_evidence(task.metadata)
    run.report["runtime_module_source"] = task.runtime_module_source


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
    # Imported here so the model-free modes of this module stay NumPy-free.
    import numpy as np

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


# ----------------------------------------------------------------------
# Source-indexed observation files and provenance, separate from the
# policy math in ``policy.py``. NumPy loads lazily inside
# :func:`load_observation` so importing this module (the manifest
# validation path) stays NumPy-free for host listing/dry-run.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class InputRecord:
    source_index: int
    path: Path
    expected_sha256: str | None


def discover_inputs(path):
    path = Path(path).resolve()
    files = sorted(path.glob("*.bin")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No observation BIN files found")
    indexed = {}
    for file in files:
        if (
            not file.is_file()
            or file.suffix != ".bin"
            or not file.stem.isascii()
            or not file.stem.isdecimal()
        ):
            raise ValueError(f"Expected a numerically named observation BIN: {file}")
        index = int(file.stem)
        if index in indexed:
            raise ValueError(f"Duplicate source index {index}")
        indexed[index] = file
    directory = path if path.is_dir() else path.parent
    manifest_path = directory.parent / "runtime-input-manifest.json"
    manifest_info = None
    records = {}
    if manifest_path.is_file():
        data = manifest_path.read_bytes()
        manifest = json.loads(data)
        contract = manifest.get("input_contract", {})
        if any(
            contract.get(k) != v
            for k, v in {
                "name": "obs_history",
                "shape": [1, 270],
                "dtype": "float32",
                "bytes_per_file": 1080,
            }.items()
        ):
            raise ValueError("Observation manifest physical contract mismatch")
        entries = manifest.get("records")
        if not isinstance(entries, list) or not entries:
            raise ValueError("Manifest requires nonempty records")
        indices = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid manifest record")
            name = entry.get("file")
            index = entry.get("source_index")
            digest = entry.get("sha256")
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or type(index) is not int
                or index < 0
                or index in indices
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or entry.get("bytes") != 1080
            ):
                raise ValueError("Invalid or duplicate manifest identity/digest")
            location = (manifest_path.parent / name).resolve()
            if (
                not location.is_relative_to(manifest_path.parent.resolve())
                or location in records
            ):
                raise ValueError(
                    "Manifest paths must be unique and within its directory"
                )
            indices.add(index)
            records[location] = (index, digest)
        manifest_info = {
            "path": str(manifest_path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "source": manifest.get("source"),
            "source_sha256": manifest.get("source_sha256"),
        }
    result = []
    for index, file in sorted(indexed.items()):
        expected = None
        if manifest_info is not None:
            if file not in records or records[file][0] != index:
                raise ValueError(f"Input identity not in manifest: {file}")
            expected = records[file][1]
        result.append(InputRecord(index, file, expected))
    return tuple(result), manifest_info


def load_observation(record):
    import numpy as np

    data = record.path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != 1080:
        raise ValueError("Observation must contain exactly 1080 bytes")
    if record.expected_sha256 is not None and digest != record.expected_sha256:
        raise ValueError(f"Observation digest mismatch: {record.path}")
    values = np.frombuffer(data, dtype="<f4").astype(np.float32).reshape(1, 270)
    if not np.isfinite(values).all():
        raise ValueError("Observation contains NaN/Inf")
    return values, digest


__all__ = [
    "ASSET_ID",
    "InputRecord",
    "ModelSelection",
    "PreparedRun",
    "build_parser",
    "complete",
    "discover_inputs",
    "load_observation",
    "normalize_args",
    "note_runtime",
    "prepare",
    "print_resolution",
    "record_sample",
    "resolve_selection",
    "validate_selection",
]
