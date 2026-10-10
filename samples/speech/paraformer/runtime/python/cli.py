"""Command-line surface and per-utterance evidence records for Paraformer.

Option declarations, argument validation/normalization, the published S100
asset identities/digests and selection, the audio/manifest input records and
the model-free listing/dry-run rendering live here, together with the
evidence discipline the entry drives: :func:`prepare` collects
input digests and creates the (new) output directory, the per-utterance
helpers — :func:`note_runtime`, :func:`prepare_utterance`,
:func:`mark_attempted`, :func:`record_prediction`, :func:`save_features` —
and :func:`complete`/:func:`record_failure` keep one implementation of the
report records. Nothing in this module loads an SDK, imports NumPy at module level or
processes audio features, so ``main.py`` stays a thin, readable entry: parse
arguments, run the model-free modes, construct the frontend and the
three-model bundle, run the per-utterance ``pipeline.predict`` loop visibly.
The stage/pipeline composition lives in ``pipeline.py``.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from utils.py_utils.assets import Asset, list_assets, sha256_file
from utils.py_utils.runtime_meta import metadata_evidence

# ======================================================================
# Published S100 asset identities, digests and selection.
# ======================================================================

SAMPLE_DIR = Path(__file__).resolve().parents[2]


STAGES = ("encoder", "predictor", "decoder")


FILENAMES = {
    "encoder": "s100/paraformer_large_encoder_400x560_s100.hbm",
    "predictor": "s100/paraformer_large_predictor_400x512_s100.hbm",
    "decoder": "s100/paraformer_large_decoder_400x512_s100.hbm",
}


CONTEXT = "/encoder/after_norm/Add_1_output_0"


LOCAL_DIGESTS = {
    "am.mvn": "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5",
    "paraformer_config.yaml": "1d9057edeaba9e131cb98f26011606497cf3af187d8943525ddb5ee36c836b1b",
}


VOCABULARY_DIGEST = "2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127"


@dataclass(frozen=True)
class Selection:
    target: str
    stage: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def resolve_selections(target="auto", *, model_paths=None, asset_ids=None):
    """Select all three models; alternate paths require all three asset IDs."""
    if target == "auto":
        from utils.py_utils.platforms import detect_target

        target = detect_target()
    if target != "s100":
        raise ValueError("Paraformer has a published three-model set only for s100")
    for name, values in (("model_paths", model_paths), ("asset_ids", asset_ids)):
        if values is not None and (
            not isinstance(values, Mapping) or set(values) != set(STAGES)
        ):
            raise ValueError(f"{name} requires exactly encoder, predictor and decoder")
    if model_paths is not None and asset_ids is None:
        raise ValueError("External model paths require explicit matching asset IDs")
    assets = {asset.filename: asset for asset in list_assets("s", "paraformer")}
    selected = []
    for stage in STAGES:
        asset = assets.get(FILENAMES[stage])
        if asset is None or asset.format != "hbm" or not asset.url:
            raise ValueError(f"Missing published {stage} HBM asset")
        if asset_ids is not None and asset_ids[stage] != asset.reference:
            raise ValueError(f"Expected {stage} asset-id {asset.reference}")
        path = (
            Path(model_paths[stage]).expanduser()
            if model_paths is not None
            else SAMPLE_DIR / "model" / asset.filename
        )
        selected.append(Selection(target, stage, asset, path, model_paths is not None))
    return tuple(selected)

# ======================================================================
# Audio/manifest file handling, kept outside feature and inference math.
# ======================================================================

@dataclass(frozen=True)
class InputItem:
    entry: dict
    audio_path: Path


def validate_id(value):
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or value.strip() != value
        or any(char in value for char in ("/", "\\", "\0"))
    ):
        raise ValueError("utt_id must be a nonempty filename stem, not a path")
    return value


def load_manifest(path, audio_dir, max_utts=0):
    if type(max_utts) is not int or max_utts < 0:
        raise ValueError("max-utts must be nonnegative; zero means all")
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must be a nonempty JSON list")
    seen = set()
    items = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each manifest entry must be an object")
        key = validate_id(record.get("utt_id"))
        if key in seen or ("text" in record and not isinstance(record["text"], str)):
            raise ValueError(
                "Manifest IDs must be unique and reference text must be a string"
            )
        seen.add(key)
        items.append(InputItem(dict(record), Path(audio_dir) / f"{key}.wav"))
    selected = items[:max_utts] if max_utts else items
    for item in selected:
        if not item.audio_path.is_file():
            raise ValueError(f"Missing selected WAV: {item.audio_path}")
    return tuple(selected)


def read_audio(path):
    import soundfile as sf

    return sf.read(Path(path), dtype="float32")


def write_json(path, payload):
    """Atomically finish JSON inside a newly created per-run directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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


@dataclass
class Preparation:
    """Collected input evidence, utterance items and the report skeleton."""

    items: tuple
    vocabulary: object
    initial: dict
    report: dict


@dataclass
class Utterance:
    """One prepared utterance: its manifest entry, record, features and timing."""

    key: str
    entry: dict
    record: dict
    features: object
    frontend_ms: float


def prepare(args, selections) -> Preparation:
    """Collect input digests and create the output directory; load no models.

    The established evidence order is preserved: the output directory must
    be new, every input (CMVN, manifest/audio, vocabulary, models) is hashed
    before use — models are hashed before the caller constructs the runtime
    bundle — and the report skeleton is complete before any utterance is
    processed.
    """
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError(
            "Output directory must be new; existing results are never overwritten"
        )
    initial = {args.cmvn_path: sha256_file(args.cmvn_path)}
    if args.audio_file is not None:
        key = validate_id(args.audio_file.stem)
        if not args.audio_file.is_file():
            raise ValueError(f"Missing WAV: {args.audio_file}")
        items = (InputItem({"utt_id": key}, args.audio_file),)
    else:
        initial[args.manifest] = sha256_file(args.manifest)
        items = load_manifest(args.manifest, args.audio_dir, args.max_utts)
    vocabulary = None
    if not args.preprocess_only:
        # Imported here so the model-free modes of this module stay NumPy-free.
        from samples.speech.paraformer.runtime.python.pipeline import (
            validate_vocabulary,
        )

        initial[args.tokens_path] = sha256_file(args.tokens_path)
        if initial[args.tokens_path] != VOCABULARY_DIGEST:
            raise ValueError("Vocabulary differs from the pinned published token order")
        vocabulary = validate_vocabulary(
            json.loads(args.tokens_path.read_text(encoding="utf-8"))
        )
        for selection in selections:
            initial[selection.model_path] = sha256_file(selection.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "target": selections[0].target,
        "mode": "preprocess-only" if args.preprocess_only else "inference",
        "inference_executed": False,
        "inference_attempted": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": args.random_seed,
        "cmvn_sha256": initial[args.cmvn_path],
        "manifest_sha256": initial.get(args.manifest),
        "utterances": [],
        "model_set": [
            {
                "stage": s.stage,
                "asset_id": s.asset.reference,
                "path": str(s.model_path),
                "publisher_sha256": s.asset.sha256,
                "observed_sha256": initial.get(s.model_path),
            }
            for s in selections
        ],
        "tokens_sha256": initial.get(args.tokens_path),
        "scheduling": {"priority": args.priority, "bpu_cores": args.bpu_cores},
    }
    return Preparation(items, vocabulary, initial, report)


def note_runtime(report, bundle) -> None:
    """Record the bound model metadata of a loaded runtime bundle."""

    if bundle is not None:
        report["metadata"] = [
            metadata_evidence(runner.binding.metadata) for runner in bundle.runners
        ]


def prepare_utterance(args, preparation, frontend, item) -> Utterance:
    """Hash, read and frontend-prepare one utterance; build its record.

    The audio digest joins ``preparation.initial`` before the audio is used,
    ``frontend.pre_process`` is timed for the record, and the record keeps
    the manifest's reference text separately from any prediction.
    """

    key = item.entry["utt_id"]
    preparation.report["current_utterance"] = key
    digest = sha256_file(item.audio_path)
    preparation.initial[item.audio_path] = digest
    audio, rate = read_audio(item.audio_path)
    start = perf_counter()
    features = frontend.pre_process(audio, rate)
    frontend_ms = (perf_counter() - start) * 1000
    record = {
        "utt_id": key,
        "audio_path": str(item.audio_path),
        "audio_sha256": digest,
        "sample_rate": rate,
        "sample_count": features.sample_count,
        "valid_frames": features.valid_frames,
        "original_frames": features.original_frames,
        "truncated": features.truncated,
        "frontend_ms": frontend_ms,
    }
    if "text" in item.entry:
        record["reference_text"] = item.entry["text"]
    return Utterance(key, item.entry, record, features, frontend_ms)


def mark_attempted(report) -> None:
    """Mark pipeline entry before the SDK pipeline is called.

    Even a failed first model call must not be reported as unattempted, and
    execution completion becomes unknown (``None``) until a prediction
    returns.
    """

    report["inference_attempted"] = True
    if report["inference_executed"] is False:
        report["inference_executed"] = None


def record_prediction(report, utterance, prediction) -> None:
    """Record one finished prediction and mark execution completed."""

    report["inference_executed"] = True
    utterance.record.update(
        text=prediction.text,
        token_ids=list(prediction.token_ids),
        token_count=prediction.token_count,
        decoder_executed=prediction.decoder_executed,
        timings_ms=prediction.timings_ms,
    )
    report["utterances"].append(utterance.record)


def save_features(args, preparation, utterance, prepared_manifest) -> None:
    """Save one prepared feature tensor and record it in both manifests."""

    import numpy as np

    relative = f"feats/{utterance.key}.npy"
    path = args.output_dir / relative
    np.save(path, utterance.features.tensor, allow_pickle=False)
    utterance.record.update(feature_file=relative, feature_sha256=sha256_file(path))
    prepared_manifest.append(
        {
            **utterance.entry,
            "feat_length": utterance.features.valid_frames,
            "original_frames": utterance.features.original_frames,
            "truncated": utterance.features.truncated,
            "feature_file": relative,
            "feature_sha256": utterance.record["feature_sha256"],
        }
    )
    preparation.report["utterances"].append(utterance.record)


def complete(args, preparation, prepared_manifest):
    """Re-verify every input digest and write the completed result record."""

    report = preparation.report
    for path, digest in preparation.initial.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Input changed during execution: {path}")
    if args.preprocess_only:
        write_json(
            args.output_dir / "prepared-manifest.json", prepared_manifest
        )
    report.pop("current_utterance", None)
    report.update(
        status="completed", ended_utc=datetime.now(timezone.utc).isoformat()
    )
    write_json(args.output_dir / "result.json", report)
    return report


def record_failure(args, report, error) -> None:
    """Write the failure record for a failure after the report skeleton exists."""
    report.update(
        status="failed",
        error=str(error),
        error_type=type(error).__name__,
        ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    try:
        write_json(args.output_dir / "failed.json", report)
    except OSError as report_error:
        raise RuntimeError(
            f"{error}; also unable to write failure record: {report_error}"
        ) from error


__all__ = [
    "Preparation",
    "Utterance",
    "build_parser",
    "complete",
    "mark_attempted",
    "normalize_args",
    "note_runtime",
    "prepare",
    "prepare_utterance",
    "print_resolution",
    "record_failure",
    "record_prediction",
    "resolve_selections_for",
    "save_features",
]
