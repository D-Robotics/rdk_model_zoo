# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free CLI and board entrypoint for the R3D-18 sample.

Option declarations, the model-free listing/dry-run modes and the JSON report
assembly live in ``cli.py``.  This entry stays focused on the execution path:
resolve the selection, load the clip and labels, construct
``VideoClassificationTask`` and call ``predict`` once.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ``3dresnet`` starts with a digit, so package imports must go through
# importlib instead of ``from samples.vision.3dresnet...`` statements.
_cli = importlib.import_module("samples.vision.3dresnet.runtime.python.cli")
build_parser = _cli.build_parser  # re-exported: the contract checker imports it from main
parse_args = _cli.parse_args  # re-exported: existing callers import it from main
report = _cli.report
run_dry_run = _cli.run_dry_run
run_list_models = _cli.run_list_models
resolve_selection = importlib.import_module(
    "samples.vision.3dresnet.runtime.python.model_binding"
).resolve_selection


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires --target s100; no board detection performed.")
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if args.dry_run:
            return run_dry_run(selection)
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {selection.model_path}; prepare it explicitly with model/download.sh.")
        import numpy as np

        classification = importlib.import_module(
            "samples.vision.3dresnet.runtime.python.classification"
        )
        labels_module = importlib.import_module(
            "samples.vision.3dresnet.runtime.python.labels"
        )
        runner_module = importlib.import_module(
            "samples.vision.3dresnet.runtime.python.model_runner"
        )
        VideoClassificationTask = classification.VideoClassificationTask
        load_labels = labels_module.load_labels
        RuntimeModelRunner = runner_module.RuntimeModelRunner

        clip_path = Path(args.test_clip).expanduser()
        clip = np.load(clip_path, allow_pickle=False)
        labels = load_labels(args.label_file)
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = VideoClassificationTask(runner, binding, top_k=args.top_k, labels=labels)
        result = task.predict(clip)
        print(json.dumps(report(result, labels, selection, clip_path), indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
