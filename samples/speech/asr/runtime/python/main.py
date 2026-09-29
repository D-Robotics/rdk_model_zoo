# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stream independent ASR chunks and save identity-bound transcription records."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.speech.asr.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
    list_available_assets,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path")
    p.add_argument(
        "--audio-file", type=Path, default=SAMPLE_DIR / "test_data/chi_sound.wav"
    )
    p.add_argument(
        "--vocab-file", type=Path, default=SAMPLE_DIR / "test_data/vocab.json"
    )
    p.add_argument("--audio-maxlen", type=int, default=30000)
    p.add_argument("--new-rate", type=int, default=16000)
    p.add_argument("--decode-mode", choices=("ctc", "legacy"), default="ctc")
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    p.add_argument("--output-dir", type=Path, default=Path("outputs/asr"))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    report = None
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {
                            "asset_id": a.reference,
                            "target": a.filename.split("/")[0],
                            "url": a.url,
                            "sha256": a.sha256,
                        }
                        for a in list_available_assets(args.target)
                    ],
                    indent=2,
                )
            )
            return 0
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit target")
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        from samples.speech.asr.runtime.python.frontend import Config, validate_config

        config = Config(args.audio_maxlen, args.new_rate)
        validate_config(config)
        if not 0 <= args.priority <= 255 or any(c < 0 for c in args.bpu_cores):
            raise ValueError("Invalid scheduling parameters")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "model_path": str(selection.model_path),
                        "input": "float32 [1,30000]",
                        "output": "[1,T,3503]; T and dtype checked at load",
                        "decode_mode": args.decode_mode,
                        "sdk_loaded": False,
                        "downloaded": False,
                        "runtime_metadata_verified": False,
                    },
                    indent=2,
                )
            )
            return 0
        from samples._shared.platforms import require_execution_target

        require_execution_target(selection.target)
        from samples._shared.assets import sha256_file
        from samples._shared.runtime_meta import metadata_evidence
        from samples.speech.asr.runtime.python.vocabulary import load_vocabulary, SHA256
        from samples.speech.asr.runtime.python.audio_io import read_chunks
        from samples.speech.asr.runtime.python.model_runner import RuntimeModelRunner
        from samples.speech.asr.runtime.python.asr import ASR

        vocabulary = load_vocabulary(args.vocab_file)
        audio = args.audio_file.expanduser()
        audio_sha = sha256_file(audio)
        destination = args.output_dir.expanduser()
        if destination.exists() or destination.is_symlink():
            raise ValueError("Output directory must be new")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = ASR(runner, binding, vocabulary, config, decode_mode=args.decode_mode)
        report = {
            "schema": "rdk-model-zoo/asr-run/v1",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": selection.asset.sha256,
            "audio_sha256": audio_sha,
            "vocabulary_sha256": SHA256,
            "decode_mode": args.decode_mode,
            "frontend": "scipy-fourier; zscore var+1e-5; normalize-before-padding",
            "config": {
                "audio_maxlen": config.audio_maxlen,
                "new_rate": config.new_rate,
            },
            "metadata": metadata_evidence(binding.metadata),
            "chunks": [],
            "status": "running",
        }
        destination.mkdir(parents=True)
        output = destination
        for chunk in read_chunks(audio, config):
            prepared = task.pre_process(chunk.waveform, chunk.sample_rate)
            text = task.post_process(
                task.forward({binding.input_name: prepared.tensor})
            )
            report["chunks"].append(
                {
                    "index": chunk.index,
                    "source_start": chunk.source_start,
                    "source_frames": len(chunk.waveform),
                    "source_rate": chunk.sample_rate,
                    "valid_target_samples": prepared.valid_samples,
                    "text": text,
                }
            )
        if not report["chunks"]:
            raise ValueError("No audio chunks were processed")
        if sha256_file(audio) != audio_sha:
            raise ValueError("Audio file changed during streaming")
        report.update(
            status="completed", text="".join(c["text"] for c in report["chunks"])
        )
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
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
