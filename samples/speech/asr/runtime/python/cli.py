# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the ASR sample.

Option declarations, the model-free listing/dry-run rendering and the
streaming report records live here so ``main.py`` can stay a thin, readable
entry: parse arguments, construct the chunk model, call ``predict`` per
chunk, show the result. Nothing in this module decodes audio or loads a
board SDK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.py_utils.runtime_meta import metadata_evidence
from samples.speech.asr.runtime.python.asr import ChunkPrediction
from samples.speech.asr.runtime.python.audio_io import AudioChunk
from samples.speech.asr.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
)
from samples.speech.asr.runtime.python.vocabulary import SHA256


def build_parser():
    """Build the SDK-free command line parser for the sample entrypoint."""

    parser = argparse.ArgumentParser(
        description="Stream independent ASR chunks and save identity-bound "
                    "transcription records."
    )
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--audio-file", type=Path, default=SAMPLE_DIR / "test_data/chi_sound.wav"
    )
    parser.add_argument(
        "--vocab-file", type=Path, default=SAMPLE_DIR / "test_data/vocab.json"
    )
    parser.add_argument("--audio-maxlen", type=int, default=30000)
    parser.add_argument("--new-rate", type=int, default=16000)
    parser.add_argument("--decode-mode", choices=("ctc", "legacy"), default="legacy",
                        help="legacy reproduces the source sample output; ctc normalizes repeats and word delimiters")
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/asr"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def run_list_models(target: str) -> int:
    """Print the manifest-backed asset references for ``target`` (model-free)."""

    print(
        json.dumps(
            [
                {
                    "asset_id": a.reference,
                    "target": a.filename.split("/")[0],
                    "url": a.url,
                    "sha256": a.sha256,
                }
                for a in list_available_assets(target)
            ],
            indent=2,
        )
    )
    return 0


def run_dry_run(selection, decode_mode: str) -> int:
    """Print the resolved contract without loading a model or SDK."""

    print(
        json.dumps(
            {
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "input": "float32 [1,30000]",
                "output": "[1,T,3503]; T and dtype checked at load",
                "decode_mode": decode_mode,
                "sdk_loaded": False,
                "downloaded": False,
                "runtime_metadata_verified": False,
            },
            indent=2,
        )
    )
    return 0


def build_report(selection, config, model, *, decode_mode, audio_sha256,
                 model_sha256) -> dict:
    """Build the streaming report skeleton before any chunk is processed.

    Args:
        selection: Resolved Selection identifying target and artifact.
        config: Frontend Config used for every chunk.
        model: Constructed ASR task (``ASR.from_model``); its ``metadata``
            property supplies the bound SDK metadata evidence.
        decode_mode: ``ctc`` or ``legacy``.
        audio_sha256: Digest of the streamed audio file.
        model_sha256: Digest of the compiled model file.

    Returns:
        dict: Report skeleton; chunks are appended by record_chunk.
    """

    return {
        "schema": "rdk-model-zoo/asr-run/v1",
        "target": selection.target,
        "asset_id": selection.asset.reference,
        "model_sha256": model_sha256,
        "publisher_sha256": selection.asset.sha256,
        "audio_sha256": audio_sha256,
        "vocabulary_sha256": SHA256,
        "decode_mode": decode_mode,
        "frontend": "scipy-fourier; zscore var+1e-5; normalize-before-padding",
        "config": {
            "audio_maxlen": config.audio_maxlen,
            "new_rate": config.new_rate,
        },
        "metadata": metadata_evidence(model.metadata),
        "chunks": [],
        "status": "running",
    }


def record_chunk(report: dict, chunk: AudioChunk, prediction: ChunkPrediction) -> None:
    """Append one chunk's record from that chunk's own prediction details."""

    report["chunks"].append(
        {
            "index": chunk.index,
            "source_start": chunk.source_start,
            "source_frames": len(chunk.waveform),
            "source_rate": chunk.sample_rate,
            "valid_target_samples": prediction.prepared.valid_samples,
            "text": prediction.text,
        }
    )


def complete_report(report: dict, output_dir: Path) -> None:
    """Mark the report completed and write ``result.json``."""

    report.update(
        status="completed", text="".join(c["text"] for c in report["chunks"])
    )
    (Path(output_dir) / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )


__all__ = [
    "build_parser",
    "build_report",
    "complete_report",
    "record_chunk",
    "run_dry_run",
    "run_list_models",
]
