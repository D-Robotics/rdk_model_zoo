# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free CLI boundary and board execution entrypoint for LPRNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Make direct ``python /abs/path/main.py`` work from any cwd without importing
# the board SDK or changing the host-safe list/dry-run paths.
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from samples.vision.lprnet.runtime.python.lprnet import LPRNetTask
from samples.vision.lprnet.runtime.python.model_binding import BindingError, SUPPORTED_TARGETS, list_available_assets, resolve_selection
from samples.vision.lprnet.runtime.python.model_runner import RuntimeModelRunner


_SAMPLE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEST_BIN = _SAMPLE_DIR / "test_data" / "test_input.dat"


def build_parser() -> argparse.ArgumentParser:
    """Build the host-safe LPRNet CLI parser."""

    parser = argparse.ArgumentParser(description="LPRNet license-plate recognition")
    parser.add_argument("--target", choices=("auto",) + SUPPORTED_TARGETS, default="auto")
    parser.add_argument("--asset-id", help="Exact manifest reference, for example x5:lprnet:lpr.bin")
    parser.add_argument("--model-path", help="Existing model path; requires the exact --asset-id")
    parser.add_argument("--test-bin", default=str(DEFAULT_TEST_BIN), help="Packed float32 input .dat path")
    parser.add_argument("--priority", type=int, default=5)
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def _validate_scheduling(args: argparse.Namespace) -> None:
    """Reject scheduling values the run would refuse, including during dry-run."""

    if type(args.priority) is not int or not 0 <= args.priority <= 255:
        raise ValueError("priority must be an integer between 0 and 255")
    if not args.bpu_cores or any(type(core) is not int or core < 0 for core in args.bpu_cores):
        raise ValueError("bpu-cores must be a non-empty list of non-negative integer indexes")


def _list_models(target: str) -> int:
    rows = list_available_assets(target)
    print(json.dumps([
        {"asset_id": row.reference, "filename": row.filename, "format": row.format,
         "url": row.url, "sha256": row.sha256, "target": "x5"}
        for row in rows
    ], ensure_ascii=False, indent=2))
    return 0


def _dry_run(args: argparse.Namespace) -> int:
    selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
    print(json.dumps({
        "target": selection.target, "asset_id": selection.asset.reference,
        "model_path": str(selection.model_path), "model_format": selection.asset.format,
        "input_shape": [1, 3, 24, 94], "input_dtype": "float32",
        "output_shape": [1, 68, 18], "output_dtype": "float32",
        "source_input": "prepacked float32 .dat; no image preprocessing",
        "model_path_exists": selection.model_path.is_file(),
        "sdk_loaded": False, "downloaded": False,
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run list, dry-run, or board inference and return 0/2."""

    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return _list_models(args.target)
        _validate_scheduling(args)
        if args.dry_run:
            return _dry_run(args)
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")
        from samples._shared.platforms import require_execution_target
        require_execution_target(selection.target)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = LPRNetTask(runner, binding)
        plate = task.predict(args.test_bin)
        print(json.dumps({"target": selection.target, "asset_id": selection.asset.reference, "plate": plate}, ensure_ascii=False))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
