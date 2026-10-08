# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the KWS sample.

Option declarations, the model-free listing/dry-run rendering and the
probability report (records, file writing, printing) live here so
``main.py`` can stay a thin, readable entry: parse arguments, construct the
runner and the ``KWS`` task, call ``predict`` once, present the report.
Nothing in this module imports NumPy, decodes audio or loads a board SDK, so
host listing, help and dry-run stay light.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json

from utils.py_utils.assets import sha256_file
from utils.py_utils.runtime_meta import metadata_evidence
from samples.speech.kws.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
)


def build_parser():
    """Build the SDK-free command line parser for the sample entrypoint."""

    parser = argparse.ArgumentParser(
        description="Canonical KWS command: explicit preparation and a "
                    "probability report."
    )
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--audio-file", type=Path, default=SAMPLE_DIR / "test_data/sample.wav"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/kws"))
    parser.add_argument("--audio-maxlen", type=int, default=60000)
    parser.add_argument("--frame-shift", type=int, default=10)
    parser.add_argument("--frame-length", type=int, default=25)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    parser.add_argument("--threshold", type=float, default=0.5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def run_list_models(target: str) -> int:
    """Print the manifest-backed asset references for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {
                    "target": "s100",
                    "asset_id": a.reference,
                    "url": a.url,
                    "sha256": a.sha256,
                }
                for a in list_available_assets(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection) -> int:
    """Print the resolved contract without loading a model or SDK."""

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


def build_report(selection, config, model, *, score, threshold, audio_file,
                 audio, rate) -> dict:
    """Build the probability report after one finished prediction.

    Args:
        selection: Resolved Selection identifying target and artifact.
        config: Fixed frontend Config validated before execution.
        model: Constructed KWS task (``KWS.from_model``); its ``metadata``
            property supplies the bound SDK metadata evidence.
        score: Maximum keyword probability returned by ``model.predict``.
        threshold: Detection threshold from the command line.
        audio_file: Audio path whose digest is recorded.
        audio: Waveform array produced by ``load_audio``.
        rate: Audio sample rate produced by ``load_audio``.

    Returns:
        dict: Complete report; written by write_report.

    Notes:
        Input digests are computed here, at the same point of the
        established execution order (after the single ``predict`` call), so
        the report never causes a second execution.
    """

    return {
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_sha256": sha256_file(selection.model_path),
        "publisher_sha256": selection.asset.sha256,
        "audio_sha256": sha256_file(audio_file),
        "sample_rate": rate,
        "source_samples": len(audio),
        "used_samples": min(len(audio), 60000),
        "padded_samples": max(0, 60000 - len(audio)),
        "truncated_samples": max(0, len(audio) - 60000),
        "config": asdict(config),
        "score": score,
        "threshold": threshold,
        "detected": score >= threshold,
        "threshold_rule": "score >= threshold",
        "metadata": metadata_evidence(model.metadata),
    }


def write_report(report: dict, output_dir: Path) -> None:
    """Create the (new) output directory, save ``result.json`` and print it."""

    output_dir.mkdir(parents=True)
    text = json.dumps(report, indent=2)
    (output_dir / "result.json").write_text(text + "\n")
    print(text)


__all__ = [
    "build_parser",
    "build_report",
    "run_dry_run",
    "run_list_models",
    "write_report",
]
