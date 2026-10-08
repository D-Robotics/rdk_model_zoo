"""Paraformer: explicit model selection, host feature preparation and inference.

This file stays deliberately small and keeps the loop visible: parse the
arguments, validate them, handle the model-free listing/dry-run modes, gate
the board, collect input evidence, construct the frontend and the
three-model pipeline, run the per-utterance loop — one
``model.predict`` per utterance between the CLI's
prepare/record helpers — and print the report. Option declarations,
validation, the model-free rendering and the evidence records live in
``cli.py``; the readable encoder → predictor → CIF → decoder composition
lives in ``pipeline.py``.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.speech.paraformer.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: existing callers import it from main
    complete,
    mark_attempted,
    normalize_args,
    note_runtime,
    prepare,
    prepare_utterance,
    print_resolution,
    record_failure,
    record_prediction,
    resolve_selections_for,
    save_features,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        normalize_args(args)
        selections = resolve_selections_for(args)
        if args.list_models or args.dry_run:
            print_resolution(args, selections)
            return 0

        # A real execution must prove the exact detected board before the SDK
        # is imported; the stage runners repeat this check immediately before
        # load.  Preprocess-only never touches an SDK and skips the gate.
        if not args.preprocess_only:
            from utils.py_utils.platforms import require_execution_target

            require_execution_target(selections[0].target)
        preparation = prepare(args, selections)

        # Imported inside real execution: Torch (frontend) and hbm_runtime
        # (pipeline) load only after the selection and board checks passed.
        try:
            from samples.speech.paraformer.runtime.python.frontend import (
                ParaformerFrontend,
            )
            from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline

            frontend = ParaformerFrontend(args.cmvn_path, random_seed=args.random_seed)
            model = None
            if not args.preprocess_only:
                model = ParaformerPipeline.from_models(selections, preparation.vocabulary)
                model.set_scheduling_params(
                    priority=args.priority, bpu_cores=args.bpu_cores
                )
        except Exception as error:
            record_failure(args, preparation.report, error)
            raise

        report = preparation.report
        try:
            note_runtime(report, model)
            prepared_manifest = []
            if args.preprocess_only:
                (args.output_dir / "feats").mkdir()
            for item in preparation.items:
                utterance = prepare_utterance(
                    args, preparation, frontend, item
                )
                if args.preprocess_only:
                    save_features(
                        args, preparation, utterance, prepared_manifest
                    )
                else:
                    # Mark attempted execution before entering the SDK
                    # pipeline so even a failed first model call is not
                    # reported as unattempted.
                    mark_attempted(report)
                    prediction = model.predict(
                        utterance.features.tensor, utterance.features.valid_frames
                    )
                    record_prediction(report, utterance, prediction)
            report = complete(args, preparation, prepared_manifest)
        except Exception as error:
            record_failure(args, report, error)
            raise
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
