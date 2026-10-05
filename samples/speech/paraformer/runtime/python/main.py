"""Paraformer: explicit model selection, host feature preparation and inference.

This file stays deliberately small: parse the arguments, validate them,
handle the model-free listing/dry-run modes, gate the board, collect input
evidence, construct the frontend and the three-model runtime bundle, run
the per-utterance loop visibly — one ``bundle.pipeline.predict`` per
utterance between the application's prepare/record helpers — and print the
report. Option declarations, validation and the model-free rendering live
in ``cli.py``; evidence collection and the per-utterance records live in
``application.py``; the readable encoder → predictor → CIF → decoder
composition lives in ``pipeline.py``.
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.speech.paraformer.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: existing callers import it from main
    normalize_args,
    print_resolution,
    resolve_selections_for,
)
from samples.speech.paraformer.runtime.python.model_binding import (  # noqa: E402
    STAGES,  # noqa: F401 - import path kept for existing callers
    resolve_selections,  # noqa: F401 - import path kept for existing callers
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
        from samples.speech.paraformer.runtime.python import application

        if not args.preprocess_only:
            from samples._shared.platforms import require_execution_target

            require_execution_target(selections[0].target)
        preparation = application.prepare(args, selections)

        # Imported inside real execution: Torch (frontend) and hbm_runtime
        # (bundle) load only after the selection and board checks passed.
        try:
            from samples.speech.paraformer.runtime.python.frontend import (
                ParaformerFrontend,
            )
            from samples.speech.paraformer.runtime.python.runtime import load_runtime

            frontend = ParaformerFrontend(args.cmvn_path, random_seed=args.random_seed)
            bundle = None
            if not args.preprocess_only:
                bundle = load_runtime(selections, preparation.vocabulary)
                bundle.set_scheduling_params(
                    priority=args.priority, bpu_cores=args.bpu_cores
                )
        except Exception as error:
            application.record_failure(args, preparation.report, error)
            raise

        report = preparation.report
        try:
            application.note_runtime(report, bundle)
            prepared_manifest = []
            if args.preprocess_only:
                (args.output_dir / "feats").mkdir()
            for item in preparation.items:
                utterance = application.prepare_utterance(
                    args, preparation, frontend, item
                )
                if args.preprocess_only:
                    application.save_features(
                        args, preparation, utterance, prepared_manifest
                    )
                else:
                    # Mark attempted execution before entering the SDK
                    # pipeline so even a failed first model call is not
                    # reported as unattempted.
                    application.mark_attempted(report)
                    prediction = bundle.pipeline.predict(
                        utterance.features.tensor, utterance.features.valid_frames
                    )
                    application.record_prediction(report, utterance, prediction)
            report = application.complete(args, preparation, prepared_manifest)
        except Exception as error:
            application.record_failure(args, report, error)
            raise
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
