# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Command-line surface for the ASR sample.

Everything around selection and delivery lives here: the published S100/S600
asset identities and resolver, option declarations, the model-free
listing/dry-run rendering, bounded audio-file streaming, the published
vocabulary loading and the streaming report records, so ``main.py`` can stay
a thin, readable entry: parse arguments, construct the chunk model, call
``predict`` per chunk, show the result. Nothing in this module decodes audio
or loads a board SDK; the frontend, decoders, tensor binding and task stages
live in ``asr.py`` (imported lazily where the streaming helpers need them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from utils.py_utils.assets import Asset, list_assets
from utils.py_utils.runtime_meta import metadata_evidence

if TYPE_CHECKING:
    # Presentation-hint only; importing the model module here would create an
    # eager cli->asr cycle the streaming helpers deliberately avoid.
    from samples.speech.asr.runtime.python.asr import ChunkPrediction

# ======================================================================
# Published S100/S600 ASR identity and selection; no SDK import.
# ======================================================================

SAMPLE_DIR = Path(__file__).resolve().parents[2]


TARGETS = ("s100", "s600")


@dataclass(frozen=True)
class Selection:
    target: str
    asset: Asset
    model_path: Path
    explicit_model_path: bool = False


def list_available_assets(target="auto"):
    if target not in ("auto", "s100", "s100p", "s600", "x5"):
        raise ValueError(f"Unknown target {target!r}")
    assets = tuple(list_assets("s", "asr"))
    if {a.reference for a in assets} != {f"s:asr:{t}/asr.hbm" for t in TARGETS}:
        raise ValueError("Expected exact S100/S600 ASR publications")
    return tuple(
        a for a in assets if target == "auto" or a.filename.startswith(target + "/")
    )


def resolve_selection(target="auto", *, asset_id=None, model_path=None):
    if target == "auto":
        from utils.py_utils.platforms import detect_target

        target = detect_target()
    if target not in TARGETS:
        raise ValueError("ASR is published only for s100 and s600")
    asset = list_available_assets(target)[0]
    if asset_id is not None and asset_id != asset.reference:
        raise ValueError(f"Expected asset-id {asset.reference}")
    if model_path is not None and asset_id is None:
        raise ValueError("An external model path requires the exact --asset-id")
    return Selection(
        target,
        asset,
        (
            Path(model_path).expanduser()
            if model_path
            else SAMPLE_DIR / "model" / asset.filename
        ),
        model_path is not None,
    )

# ======================================================================
# Bounded audio-file streaming outside ASR task math.
# ======================================================================

@dataclass(frozen=True)
class AudioChunk:
    waveform: np.ndarray
    sample_rate: int
    source_start: int
    index: int


def read_chunks(path, config=None):
    from samples.speech.asr.runtime.python.asr import Config, source_chunk_size

    if config is None:
        config = Config()
    import soundfile as sf

    with sf.SoundFile(Path(path).expanduser(), "r") as stream:
        rate = int(stream.samplerate)
        size = source_chunk_size(rate, config)
        if stream.frames <= 0:
            raise ValueError("Audio file contains no frames")
        start = 0
        index = 0
        while True:
            data = stream.read(size, dtype="float32")
            if len(data) == 0:
                break
            yield AudioChunk(data, rate, start, index)
            start += len(data)
            index += 1

# ======================================================================
# The exact published 3503-token mapping, outside inference stages.
# ======================================================================

SHA256 = "33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7"


def load_vocabulary(path):
    raw = Path(path).expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Vocabulary bytes differ from the published ASR token mapping")
    mapping = json.loads(raw)
    if (
        not isinstance(mapping, dict)
        or len(mapping) != 3503
        or any(type(i) is not int for i in mapping.values())
        or set(mapping.values()) != set(range(3503))
    ):
        raise ValueError("Vocabulary IDs must be unique and contiguous from 0 to 3502")
    tokens = [""] * 3503
    for token, i in mapping.items():
        tokens[i] = token
    from samples.speech.asr.runtime.python.asr import validate_vocabulary

    return validate_vocabulary(tokens)


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


def record_chunk(report: dict, chunk: "AudioChunk", prediction: "ChunkPrediction") -> None:
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
    "AudioChunk",
    "SHA256",
    "Selection",
    "build_parser",
    "build_report",
    "complete_report",
    "list_available_assets",
    "load_vocabulary",
    "read_chunks",
    "record_chunk",
    "resolve_selection",
    "run_dry_run",
    "run_list_models",
]
