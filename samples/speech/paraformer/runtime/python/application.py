"""One explicit audio/manifest operation with owned outputs and failure records."""

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from samples._shared.assets import sha256_file
from samples._shared.runtime_meta import metadata_evidence
from samples.speech.paraformer.runtime.python import input_io
from samples.speech.paraformer.runtime.python.decoding import validate_vocabulary
from samples.speech.paraformer.runtime.python.model_binding import VOCABULARY_DIGEST


def execute(args, selections):
    """Run frontend-only preparation or board-gated SDK inference; never download."""
    from samples.speech.paraformer.runtime.python.frontend import ParaformerFrontend
    from samples.speech.paraformer.runtime.python.runtime import load_runtime

    if not args.preprocess_only:
        from samples._shared.platforms import require_execution_target

        require_execution_target(selections[0].target)
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
    try:
        frontend = ParaformerFrontend(args.cmvn_path, random_seed=args.random_seed)
        bundle = None
        if not args.preprocess_only:
            bundle = load_runtime(selections, vocabulary)
            bundle.set_scheduling_params(
                priority=args.priority, bpu_cores=args.bpu_cores
            )
            report["metadata"] = [
                metadata_evidence(runner.binding.metadata) for runner in bundle.runners
            ]
        prepared_manifest = []
        if args.preprocess_only:
            (args.output_dir / "feats").mkdir()
        for item in items:
            key = item.entry["utt_id"]
            report["current_utterance"] = key
            digest = sha256_file(item.audio_path)
            initial[item.audio_path] = digest
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
            if args.preprocess_only:
                relative = f"feats/{key}.npy"
                path = args.output_dir / relative
                np.save(path, features.tensor, allow_pickle=False)
                record.update(feature_file=relative, feature_sha256=sha256_file(path))
                prepared_manifest.append(
                    {
                        **item.entry,
                        "feat_length": features.valid_frames,
                        "original_frames": features.original_frames,
                        "truncated": features.truncated,
                        "feature_file": relative,
                        "feature_sha256": record["feature_sha256"],
                    }
                )
            else:
                # Mark attempted execution before entering the SDK pipeline so
                # even a failed first model call is not reported as unattempted.
                report["inference_attempted"] = True
                if report["inference_executed"] is False:
                    report["inference_executed"] = None
                prediction = bundle.pipeline.predict(
                    features.tensor, features.valid_frames
                )
                report["inference_executed"] = True
                record.update(
                    text=prediction.text,
                    token_ids=list(prediction.token_ids),
                    token_count=prediction.token_count,
                    decoder_executed=prediction.decoder_executed,
                    timings_ms=prediction.timings_ms,
                )
            report["utterances"].append(record)
        for path, digest in initial.items():
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
    except Exception as error:
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
        raise
