# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stream independent ASR chunks and save identity-bound transcription records.

This file stays deliberately small: parse the arguments, handle the
model-free listing/dry-run modes, gate the board, construct the chunk model,
call ``predict`` once per chunk, save the report. Option declarations, the
model-free rendering, the report records, audio streaming and vocabulary
loading live in ``cli.py``; the frontend, decoders, tensor binding, the
readable preprocess → infer → postprocess chain and the raw runner
construction live in ``asr.py``.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.speech.asr.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    build_report,
    complete_report,
    record_chunk,
    resolve_selection,
    run_dry_run,
    run_list_models,
)



def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    report = None
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit target")
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        from samples.speech.asr.runtime.python.asr import Config, validate_config

        config = Config(args.audio_maxlen, args.new_rate)
        validate_config(config)
        if not 0 <= args.priority <= 255 or any(c < 0 for c in args.bpu_cores):
            raise ValueError("Invalid scheduling parameters")
        if args.dry_run:
            return run_dry_run(selection, args.decode_mode)
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        from utils.py_utils.assets import sha256_file
        from samples.speech.asr.runtime.python.asr import ASR
        from samples.speech.asr.runtime.python.cli import (
            load_vocabulary,
            read_chunks,
        )

        vocabulary = load_vocabulary(args.vocab_file)
        audio = args.audio_file.expanduser()
        audio_sha = sha256_file(audio)
        destination = args.output_dir.expanduser()
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output directory must be new")
        task = ASR.from_model(
            selection, vocabulary, config, decode_mode=args.decode_mode
        )
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        report = build_report(
            selection,
            config,
            task,
            decode_mode=args.decode_mode,
            audio_sha256=audio_sha,
            model_sha256=sha256_file(selection.model_path),
        )
        destination.mkdir(parents=True)
        output = destination
        for chunk in read_chunks(audio, config):
            # One predict per chunk: the report records this call's own
            # resampling geometry from the same single model execution.
            prediction = task.predict(
                chunk.waveform, chunk.sample_rate, return_details=True
            )
            record_chunk(report, chunk, prediction)
        if not report["chunks"]:
            raise ValueError("No audio chunks were processed")
        if sha256_file(audio) != audio_sha:
            raise ValueError("Audio file changed during streaming")
        complete_report(report, output)
        print(report["text"])
        print(f'Report: {output / "result.json"}')
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        if output is not None and report is not None:
            report.update(status="failed", error=str(error))
            try:
                (output / "failed.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )
            except OSError as report_error:
                print(f"Could not save failure report: {report_error}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
