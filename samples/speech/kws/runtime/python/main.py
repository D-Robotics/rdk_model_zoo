# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Canonical KWS command: explicit preparation and a probability report.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, gate the board, construct the ``KWS``
task through ``KWS.from_model``, apply scheduling, call ``predict`` once,
present the report. Option declarations, published selection, the model-free rendering, audio
loading and the report records live in ``cli.py``; the MDTC frontend, the
readable preprocess → infer → postprocess chain, the raw runner construction
and the model-owned loader live in ``kws.py``.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.speech.kws.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    build_report,
    resolve_selection,
    run_dry_run,
    run_list_models,
    write_report,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.audio_file = args.audio_file.expanduser()
    args.output_dir = args.output_dir.expanduser()
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit target")
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        from samples.speech.kws.runtime.python.kws import Config, validate_config

        config = Config(
            args.audio_maxlen, args.frame_shift, args.frame_length, args.n_mels
        )
        validate_config(config)
        if not 0 <= args.priority <= 255 or any(i < 0 for i in args.bpu_cores):
            raise ValueError("Invalid scheduling parameters")
        if not 0 <= args.threshold <= 1:
            raise ValueError("threshold must be finite in [0,1]")
        if args.dry_run:
            return run_dry_run(selection)
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        from samples.speech.kws.runtime.python.cli import load_audio
        from samples.speech.kws.runtime.python.kws import KWS

        audio, rate = load_audio(args.audio_file)
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError("Output directory must be new")
        task = KWS.from_model(selection, config)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        score = task.predict(audio, rate)
        report = build_report(
            selection,
            config,
            task,
            score=score,
            threshold=args.threshold,
            audio_file=args.audio_file,
            audio=audio,
            rate=rate,
        )
        write_report(report, args.output_dir)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
