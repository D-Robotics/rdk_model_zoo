# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Planning CLI entry: construct the task, run one ``predict``, archive evidence.

Option declarations, the model-free listing/dry-run modes and the canonical
output/report writing live in ``cli.py``.  This entry stays focused on the
execution path: resolve the selection, load the features and runner,
construct ``DiffusionDriveTask`` and call ``predict`` once.  The canonical
outputs include ``physical_inputs.npz`` and ``raw_outputs.npz``, so the entry
requests this call's raw IO through the opt-in ``return_details`` record
instead of recomputing stages.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import sha256_file  # noqa: F401 - re-export used by callers
from samples.vision.diffusiondrive.runtime.python.cli import (
    add_runtime_arguments,  # re-exported here: run_cases.py builds its parser from it
    build_parser,  # re-exported here: the contract checker imports it from main
    resolve_extras,
    run_dry_run,
    run_list_models,
    save_planning_evidence,
    validate_extra_destinations,
)
from samples.vision.diffusiondrive.runtime.python.data_io import (
    load_features,
    validate_destinations,
)
from samples.vision.diffusiondrive.runtime.python.model_binding import (
    resolve_selection,
)
from samples.vision.diffusiondrive.runtime.python.model_runner import RuntimeModelRunner


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        if (
            not np.isfinite(args.agent_score_thres)
            or not 0 <= args.agent_score_thres <= 1
        ):
            raise ValueError("Agent score threshold must be finite and within [0,1]")
        if not 0 <= args.priority <= 255 or any(c < 0 for c in args.bpu_cores):
            raise ValueError("Priority must be 0..255; core IDs must be nonnegative")
        if args.dry_run:
            return run_dry_run(selection, args)
        extras = resolve_extras(args)
        output = validate_destinations(args.output, extras.values())
        validate_extra_destinations(extras)
        input_path = args.input_npz.expanduser().resolve()
        features = load_features(input_path)
        started = datetime.now(timezone.utc).isoformat()
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        from samples.vision.diffusiondrive.runtime.python.diffusiondrive import (
            DiffusionDriveTask,
        )

        task = DiffusionDriveTask(runner, binding, args.agent_score_thres)
        details = task.predict(features, return_details=True)
        finished = datetime.now(timezone.utc).isoformat()
        save_planning_evidence(
            output,
            extras,
            selection=selection,
            runner=runner,
            binding=binding,
            args=args,
            input_path=input_path,
            features=features,
            details=details,
            started=started,
            finished=finished,
        )
        print(f"Saved trajectory, agents, BEV and raw IO to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
