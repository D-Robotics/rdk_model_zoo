"""Command-line surface for the Paraformer sample.

Option declarations, argument validation/normalization, selection
resolution and the model-free listing/dry-run rendering live here so
``main.py`` can stay a thin, readable entry: parse arguments, run the
model-free modes, construct the frontend and the three-model bundle, run
the application loop.  Nothing in this module loads an SDK or processes
audio.
"""

import argparse
import json
from pathlib import Path

from samples.speech.paraformer.runtime.python.model_binding import (
    SAMPLE_DIR,
    STAGES,
    resolve_selections,
)


def build_parser():
    """Build the SDK-free parser for the Paraformer entry."""
    # Keep the established program description byte-identical.
    parser = argparse.ArgumentParser(
        description="Paraformer: explicit model selection, host feature "
                    "preparation and inference."
    )
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--preprocess-only", action="store_true")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--audio-file", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paraformer"))
    parser.add_argument("--max-utts", type=int, default=0)
    parser.add_argument("--cmvn-path", type=Path, default=SAMPLE_DIR / "model/am.mvn")
    parser.add_argument(
        "--tokens-path", type=Path, default=SAMPLE_DIR / "model/s100/tokens.json"
    )
    parser.add_argument("--random-seed", type=int, default=191009)
    parser.add_argument("--priority", type=int)
    parser.add_argument("--bpu-cores", type=int, nargs="+")
    for stage in STAGES:
        parser.add_argument(f"--{stage}-model-path", type=Path)
        parser.add_argument(f"--{stage}-asset-id")
    return parser


def normalize_args(args) -> None:
    """Validate flag combinations and expand every path argument in place.

    Mutually exclusive modes, scheduling flags and the partial model
    path/asset groups are rejected here, before any selection resolution.
    """
    if args.max_utts < 0 or not 0 <= args.random_seed < 2**63:
        raise ValueError(
            "max-utts must be nonnegative and seed must be in [0,2**63)"
        )
    if args.priority is not None and not 0 <= args.priority <= 255:
        raise ValueError("priority must be in [0,255]")
    if args.bpu_cores is not None and any(core < 0 for core in args.bpu_cores):
        raise ValueError("bpu-cores must be nonnegative")
    if args.preprocess_only and (
        args.priority is not None or args.bpu_cores is not None
    ):
        raise ValueError("Scheduling flags do not apply to preprocess-only")
    if args.audio_file is not None and args.audio_dir is not None:
        raise ValueError("audio-dir is only meaningful with a manifest")
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.manifest is None and args.audio_file is None:
        args.manifest = SAMPLE_DIR / "test_data/manifest.json"
    if args.audio_dir is None and args.manifest is not None:
        args.audio_dir = args.manifest.parent / "audio"
    paths = {stage: getattr(args, f"{stage}_model_path") for stage in STAGES}
    ids = {stage: getattr(args, f"{stage}_asset_id") for stage in STAGES}
    for label, values in (("model paths", paths), ("asset IDs", ids)):
        if any(v is not None for v in values.values()) and any(
            v is None for v in values.values()
        ):
            raise ValueError(f"Provide all three {label}, not a partial group")


def resolve_selections_for(args):
    """Resolve the three stage selections for the parsed arguments.

    The model-free listing and preprocess-only modes default an ``auto``
    target to the published S100 set; host dry-run requires the explicit
    target because no board detection happens there.
    """
    target = args.target
    if target == "auto" and (args.list_models or args.preprocess_only):
        target = "s100"
    if target == "auto" and args.dry_run:
        raise ValueError("Host dry-run requires --target s100")
    paths = {stage: getattr(args, f"{stage}_model_path") for stage in STAGES}
    ids = {stage: getattr(args, f"{stage}_asset_id") for stage in STAGES}
    return resolve_selections(
        target,
        model_paths=paths if paths["encoder"] else None,
        asset_ids=ids if ids["encoder"] else None,
    )


def print_resolution(args, selections) -> None:
    """Render the model-free listing/dry-run report (no SDK, no downloads)."""
    print(
        json.dumps(
            {
                "target": selections[0].target,
                "models": [
                    {
                        "stage": s.stage,
                        "asset_id": s.asset.reference,
                        "model_path": str(s.model_path),
                        "url": s.asset.url,
                        "publisher_sha256": s.asset.sha256,
                    }
                    for s in selections
                ],
                "sdk_loaded": False,
                "downloaded": False,
                "runtime_metadata_verified": False,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


__all__ = ["build_parser", "normalize_args", "print_resolution", "resolve_selections_for"]
