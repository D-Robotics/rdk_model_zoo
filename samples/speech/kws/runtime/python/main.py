# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Canonical KWS command: explicit preparation and a probability report."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.speech.kws.runtime.python.model_binding import (
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
        "--audio-file", type=Path, default=SAMPLE_DIR / "test_data/sample.wav"
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/kws"))
    p.add_argument("--audio-maxlen", type=int, default=60000)
    p.add_argument("--frame-shift", type=int, default=10)
    p.add_argument("--frame-length", type=int, default=25)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    p.add_argument("--threshold", type=float, default=0.5)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.audio_file = args.audio_file.expanduser()
    args.output_dir = args.output_dir.expanduser()
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {
                            "target": "s100",
                            "asset_id": a.reference,
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
        from samples.speech.kws.runtime.python.frontend import Config, validate_config

        config = Config(
            args.audio_maxlen, args.frame_shift, args.frame_length, args.n_mels
        )
        validate_config(config)
        if not 0 <= args.priority <= 255 or any(i < 0 for i in args.bpu_cores):
            raise ValueError("Invalid scheduling parameters")
        if not 0 <= args.threshold <= 1:
            raise ValueError("threshold must be finite in [0,1]")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "model_path": str(selection.model_path),
                        "input": "float32 [1,373,80]",
                        "audio_samples": 60000,
                        "sample_rate": 16000,
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
        from samples.speech.kws.runtime.python.audio_io import load_audio
        from samples.speech.kws.runtime.python.model_runner import RuntimeModelRunner
        from samples.speech.kws.runtime.python.kws import KWS
        from samples._shared.assets import sha256_file
        from samples._shared.runtime_meta import metadata_evidence
        from dataclasses import asdict

        audio, rate = load_audio(args.audio_file)
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError("Output directory must be new")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        score = KWS(runner, binding, config).predict(audio, rate)
        report = {
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": selection.asset.sha256,
            "audio_sha256": sha256_file(args.audio_file),
            "sample_rate": rate,
            "source_samples": len(audio),
            "used_samples": min(len(audio), 60000),
            "padded_samples": max(0, 60000 - len(audio)),
            "truncated_samples": max(0, len(audio) - 60000),
            "config": asdict(config),
            "score": score,
            "threshold": args.threshold,
            "detected": score >= args.threshold,
            "threshold_rule": "score >= threshold",
            "metadata": metadata_evidence(binding.metadata),
        }
        args.output_dir.mkdir(parents=True)
        text = json.dumps(report, indent=2)
        (args.output_dir / "result.json").write_text(text + "\n")
        print(text)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
