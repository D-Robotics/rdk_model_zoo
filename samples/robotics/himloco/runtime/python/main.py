"""HIMLoco: published X5 model selection and source-indexed offline inference.

This entry stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, resolve the selection, then run the
offline loop visibly — obtain the bound policy task through
``application.load_task``, execute the explicitly requested warmups, call
``task.predict`` once per observation and record each action dump through
``application`` helpers, which also own the report/failure evidence. The
policy's preprocess → infer → postprocess chain lives in ``policy.py``.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from samples.robotics.himloco.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError("priority must be in [0,255]")
        if args.bpu_cores is not None and any(c < 0 for c in args.bpu_cores):
            raise ValueError("BPU cores must be nonnegative")
        for key, value in vars(args).items():
            if isinstance(value, Path):
                setattr(args, key, value.expanduser().resolve())
        target = "x5" if args.list_models and args.target == "auto" else args.target
        if args.dry_run and target == "auto":
            raise ValueError("Host dry-run requires --target x5")
        selected = resolve_selection(
            target, model_path=args.model_path, asset_id=args.asset_id
        )
        if args.list_models or args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selected.target,
                        "asset_id": selected.asset.reference,
                        "model_path": str(selected.model_path),
                        "url": selected.asset.url,
                        "sha256": selected.asset.sha256,
                        "input_path": str(args.input_path),
                        "sdk_loaded": False,
                        "downloaded": False,
                        "metadata_verified": False,
                    },
                    indent=2,
                )
            )
            return 0
        from samples.robotics.himloco.runtime.python import application
        from samples.robotics.himloco.runtime.python.input_io import load_observation

        run = application.prepare(args, selected)
        try:
            task = application.load_task(run)
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
                application.record_sample(run, record, result, digest)
                latencies.append(result.latency_ms)
            application.complete(run, latencies)
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
