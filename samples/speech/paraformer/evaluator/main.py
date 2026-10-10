"""Evaluate explicit prepared features with CPU FP32 or HMCT simulated models."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.py_utils.assets import sha256_file
from utils.py_utils.text_metrics import score_transcripts
from samples.speech.paraformer.evaluator.inputs import read_manifest, load_feature
from samples.speech.paraformer.runtime.python.cli import write_json
from samples.speech.paraformer.evaluator.backends import Stage, tensor_names
from samples.speech.paraformer.runtime.python.pipeline import validate_vocabulary
from samples.speech.paraformer.runtime.python.cli import STAGES, VOCABULARY_DIGEST
from samples.speech.paraformer.runtime.python.pipeline import ParaformerPipeline


def utc():
    return datetime.now(timezone.utc).isoformat()


def execute(args):
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise ValueError(
            "Output directory must be new; existing results are never overwritten"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "started_utc": utc(),
        "pipeline": args.pipeline,
        "backend": "CPU ONNX Runtime" if args.pipeline == "fp32" else "HMCT simulation",
        "board_test": "not-run",
        "threads": args.threads,
        "thread_scope": "ORT CPU only; HMCT uses its own session defaults",
        "max_utts": args.max_utts,
        "utterances": [],
        "metrics": None,
        "current_utterance": None,
        "models": {},
        "timing_scope": "stage calls and CPU CIF only; excludes frontend, loading and I/O; not BPU or end-to-end latency",
        "versions": {},
    }
    report_path = args.output_dir / "evaluation.json"
    try:
        for package in ("numpy", "onnx", "onnxruntime", "hmct"):
            try:
                report["versions"][package] = version(package)
            except PackageNotFoundError:
                report["versions"][package] = None
        entries, manifest_digest = read_manifest(args.manifest, args.max_utts)
        report["manifest"] = {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_digest,
        }
        report["selected_count"] = len(entries)
        raw_vocab = args.vocab.read_bytes()
        vocab_digest = hashlib.sha256(raw_vocab).hexdigest()
        report["vocabulary"] = {
            "path": str(args.vocab.resolve()),
            "sha256": vocab_digest,
        }
        if vocab_digest != VOCABULARY_DIGEST:
            raise ValueError("Vocabulary differs from the pinned published token order")
        vocabulary = validate_vocabulary(json.loads(raw_vocab))
        initial = {args.manifest: manifest_digest, args.vocab: vocab_digest}
        stages = {}
        for stage in STAGES:
            path = getattr(args, stage).resolve()
            digest = sha256_file(path)
            initial[path] = digest
            report["models"][stage] = {"path": str(path), "sha256": digest}
            stages[stage] = Stage(path, stage, args.pipeline, args.threads)
            report["models"][stage]["interface"] = stages[stage].metadata
        pipeline = ParaformerPipeline(
            *(stages[s].forward for s in STAGES), tensor_names(stages), vocabulary
        )
        for entry in entries:
            report["current_utterance"] = entry.source["utt_id"]
            write_json(report_path, report)
            features, digest = load_feature(entry)
            prediction = pipeline.predict(features, entry.source["feat_length"])
            report["utterances"].append(
                {
                    "id": entry.source["utt_id"],
                    "reference": entry.source["text"],
                    "hypothesis": prediction.text,
                    "source": entry.source,
                    "feature_file": str(entry.path),
                    "feature_sha256": digest,
                    "prediction": asdict(prediction),
                }
            )
        for path, digest in initial.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Input changed during evaluation: {path}")
        report["metrics"] = score_transcripts(report["utterances"])
        report["current_utterance"] = None
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_utc"] = utc()
        write_json(report_path, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--pipeline", choices=("fp32", "int16"), required=True)
    for stage, alias in (("encoder", "enc"), ("predictor", "pred"), ("decoder", "dec")):
        result.add_argument(f"--{stage}", f"--{alias}", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--vocab", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument(
        "--max-utts", type=int, default=0, help="select a prefix; 0 means all"
    )
    result.add_argument(
        "--threads",
        type=int,
        default=4,
        help="FP32 ORT intra-op threads; not an HMCT setting",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = execute(args)
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "count": report["metrics"]["count"],
                "cer": report["metrics"]["cer"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
