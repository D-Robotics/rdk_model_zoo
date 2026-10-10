"""HIMLoco: published X5 model selection and source-indexed offline inference.

This entry stays deliberately small and keeps the construction and loop
visible: parse the arguments, handle the model-free listing/dry-run modes,
resolve the selection, prepare the evidence-collected run, construct the
bound policy task through ``HimLocoTask.from_model``, apply scheduling,
record the runtime evidence, execute the explicitly requested warmups,
call ``task.predict`` once per observation, record each action dump and
complete the report through the ``cli`` helpers. The policy's preprocess →
infer → postprocess chain, the raw runner construction and the model-owned
loader live in ``policy.py``; offline policy inference only — never
actuators.
"""

import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from samples.robotics.himloco.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    complete,
    load_observation,
    normalize_args,
    note_runtime,
    prepare,
    print_resolution,
    record_sample,
    resolve_selection,
)


def main(argv=None):
    """Run the locomotion-policy CLI: resolve the published assets, run one observation sequence, and present the predicted actions."""

    args = build_parser().parse_args(argv)
    try:
        normalize_args(args)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires --target x5")
        selected = resolve_selection(
            args.target, model_path=args.model_path, asset_id=args.asset_id
        )
        if args.list_models or args.dry_run:
            return print_resolution(args, selected)
        run = prepare(args, selected)
        try:
            # Visible model construction: the task owns its runner and load.
            from samples.robotics.himloco.runtime.python.policy import HimLocoTask

            task = HimLocoTask.from_model(run.selection)
            task.set_scheduling_params(
                priority=args.priority, bpu_cores=args.bpu_cores
            )
            note_runtime(run, task)
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
        print(
            json.dumps(
                {
                    "status": run.report["status"],
                    "sample_count": run.report["sample_count"],
                    "output_dir": str(args.output_dir),
                }
            )
        )
        return 0
    except Exception as error:
        print(f"HIMLoco failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
