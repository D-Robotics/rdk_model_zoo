"""One explicit audio/manifest operation with owned outputs and failure records.

The workflow is split so the entry can show the model construction and the
per-utterance ``pipeline.predict`` calls while this module keeps the evidence
discipline: :func:`prepare` collects input digests and creates the (new)
output directory, the caller constructs the frontend and the three-model
runtime bundle, and the per-utterance helpers — :func:`note_runtime`,
:func:`prepare_utterance`, :func:`mark_attempted`, :func:`record_prediction`,
:func:`save_features` — plus :func:`complete` drive one utterance at a time
through ``frontend.pre_process`` and the caller's visible
``pipeline.predict``. :func:`run` keeps the established loop composition and
:func:`execute` the single-call composition of the same steps.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from time import perf_counter

import numpy as np

from utils.py_utils.assets import sha256_file
from utils.py_utils.runtime_meta import metadata_evidence
from samples.speech.paraformer.runtime.python import input_io
from samples.speech.paraformer.runtime.python.decoding import validate_vocabulary
from samples.speech.paraformer.runtime.python.model_binding import VOCABULARY_DIGEST


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
        key = input_io.validate_id(args.audio_file.stem)
        if not args.audio_file.is_file():
            raise ValueError(f"Missing WAV: {args.audio_file}")
        items = (input_io.InputItem({"utt_id": key}, args.audio_file),)
    else:
        initial[args.manifest] = sha256_file(args.manifest)
        items = input_io.load_manifest(args.manifest, args.audio_dir, args.max_utts)
    vocabulary = None
    if not args.preprocess_only:
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
    audio, rate = input_io.read_audio(item.audio_path)
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
        input_io.write_json(
            args.output_dir / "prepared-manifest.json", prepared_manifest
        )
    report.pop("current_utterance", None)
    report.update(
        status="completed", ended_utc=datetime.now(timezone.utc).isoformat()
    )
    input_io.write_json(args.output_dir / "result.json", report)
    return report


def run(args, selections, preparation, frontend, bundle):
    """Compatibility loop composition over the per-utterance helpers."""
    report = preparation.report
    try:
        note_runtime(report, bundle)
        prepared_manifest = []
        if args.preprocess_only:
            (args.output_dir / "feats").mkdir()
        for item in preparation.items:
            utterance = prepare_utterance(args, preparation, frontend, item)
            if args.preprocess_only:
                save_features(args, preparation, utterance, prepared_manifest)
            else:
                mark_attempted(report)
                prediction = bundle.pipeline.predict(
                    utterance.features.tensor, utterance.features.valid_frames
                )
                record_prediction(report, utterance, prediction)
        return complete(args, preparation, prepared_manifest)
    except Exception as error:
        record_failure(args, report, error)
        raise


def record_failure(args, report, error) -> None:
    """Write the failure record for a failure after the report skeleton exists."""
    report.update(
        status="failed",
        error=str(error),
        error_type=type(error).__name__,
        ended_utc=datetime.now(timezone.utc).isoformat(),
    )
    try:
        input_io.write_json(args.output_dir / "failed.json", report)
    except OSError as report_error:
        raise RuntimeError(
            f"{error}; also unable to write failure record: {report_error}"
        ) from error


def execute(args, selections):
    """Compatibility composition: gate, prepare, construct and run."""
    from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
    from samples.speech.paraformer.runtime.python.runtime import load_runtime

    if not args.preprocess_only:
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selections[0].target)
    preparation = prepare(args, selections)
    try:
        frontend = ParaformerFrontend(args.cmvn_path, random_seed=args.random_seed)
        bundle = None
        if not args.preprocess_only:
            bundle = load_runtime(selections, preparation.vocabulary)
            bundle.set_scheduling_params(
                priority=args.priority, bpu_cores=args.bpu_cores
            )
    except Exception as error:
        record_failure(args, preparation.report, error)
        raise
    return run(args, selections, preparation, frontend, bundle)
