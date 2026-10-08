# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""People and Agent entrypoint for the LPRNet recognition sample.

This file stays deliberately small: parse the arguments, resolve the model,
construct the recognizer, call ``predict``, present the result. Option
declarations and model-free listing/dry-run live in ``cli.py``; the
recognition flow lives in ``lprnet.py``.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

# Make direct ``python /abs/path/main.py`` work from any cwd without importing
# the board SDK or changing the host-safe list/dry-run paths.
_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from samples.vision.lprnet.runtime.python.cli import (  # noqa: E402
    BindingError, build_parser, resolve_selection, run_dry_run,
    run_list_models, validate_scheduling,
)


def main(argv: list[str] | None = None) -> int:
    """Run list, dry-run, or board inference and return 0/2.

    Args:
        argv: Optional command-line argument sequence, excluding the program
            name. None reads sys.argv through argparse.

    Returns:
        int: 0 for success; 2 for a reported selection, IO, or runtime error.

    Raises:
        SystemExit: argparse handles --help or rejects invalid arguments.

    Notes:
        List and dry-run modes do not load a model or the SDK.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        validate_scheduling(args)
        if args.dry_run:
            return run_dry_run(
                resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path))
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"model file not found: {selection.model_path}")

        # Real execution starts here: construction gates board identity and
        # the publication hash before the SDK import.
        from samples.vision.lprnet.runtime.python.lprnet import LPRNetRecognizer

        model = LPRNetRecognizer(selection)
        model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        plate = model.predict(args.test_bin)
        print(json.dumps({"target": selection.target, "asset_id": selection.asset.reference, "plate": plate}, ensure_ascii=False))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
