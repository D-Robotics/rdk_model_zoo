#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate INT8 calibration data for the Whisper encoder.

This script reads audio from WAV files OR Parquet datasets, computes the
80-channel normalized log-Mel spectrogram for each sample, and saves the
result as a headerless float32 ``.bin`` file in the calibration data directory.

The output files are ready to be consumed by ``hb_mapper makertbin`` with
``preprocess_on: False``.

Usage (WAV directory):
    python3 generate_calibration_data.py \\
        --audio-dir /path/to/wav/files \\
        --mel-filters ../model/mel_80_filters.txt \\
        --output-dir ../cal_data \\
        --num-samples 50

Usage (Parquet file):
    python3 generate_calibration_data.py \\
        --parquet-file /path/to/dataset.parquet \\
        --mel-filters ../model/mel_80_filters.txt \\
        --output-dir ../cal_data \\
        --num-samples 50

Notes:
    - For WAV: any sample rate is accepted (resampling to 16 kHz is automatic).
    - Parquet rows can contain Hugging Face audio fields with encoded bytes.
    - Output shape per sample: ``(1, 80, 3000, 1)`` float32.
    - Choose recordings that represent the intended deployment audio.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys

import numpy as np

# Use the same encoder features for calibration and board inference.
_SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_SAMPLE_DIR, "runtime", "python"))
import whisper_preprocess

_LOGGER = logging.getLogger("CalibrationGen")

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")


def collect_audio_files(audio_dir: str, num_samples: int) -> list[str]:
    """Collect up to num_samples audio files from a directory.

    Args:
        audio_dir: Directory containing audio files.
        num_samples: Maximum number of files to collect.

    Returns:
        Paths to matching files in sorted order.
    """
    files = []
    for filename in sorted(os.listdir(audio_dir)):
        if filename.lower().endswith(AUDIO_EXTENSIONS):
            files.append(os.path.join(audio_dir, filename))
    if num_samples > 0:
        files = files[:num_samples]
    return files


def read_parquet_audio(parquet_path: str,
                       num_samples: int) -> list[tuple[str, np.ndarray]]:
    """Read audio samples from a Parquet dataset file.

    Accepts encoded audio bytes, numeric arrays and Hugging Face audio fields.

    Args:
        parquet_path: Path to the .parquet file.
        num_samples: Maximum number of samples to extract.

    Returns:
        A list of ``(name, audio_array)`` tuples where ``audio_array`` is a
        1-D float32 numpy array of audio samples.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(
            "pandas and pyarrow are required for parquet support. "
            "Install with: pip install pandas pyarrow") from error

    df = pd.read_parquet(parquet_path)
    _LOGGER.info("Parquet file loaded: %d rows, columns: %s", len(df),
                 list(df.columns))

    if num_samples > 0 and len(df) > num_samples:
        df = df.head(num_samples)

    samples = []
    for idx, row in df.iterrows():
        try:
            audio_array = _extract_audio_from_row(idx, row)
            if audio_array is None:
                continue

            name = f"parquet_{idx:04d}"
            samples.append((name, audio_array.astype(np.float32)))

        except Exception as error:
            _LOGGER.warning("Row %s: failed to parse audio: %s", idx, error)

    return samples


def _extract_audio_from_row(idx: int, row) -> np.ndarray | None:
    """Read one recording from a nested or flattened Parquet audio field.

    Args:
        idx: Row index used in diagnostic messages.
        row: A dataset row containing audio bytes or waveform samples.

    Returns:
        The float32 waveform at 16 kHz, or None for an unsupported row.
    """
    audio_col = row.get("audio")
    sr = 16000

    if audio_col is None:
        # Maybe columns are separated: "audio.array" and "audio.sampling_rate"
        if "audio.array" in row.index:
            audio_col = list(row["audio.array"])
            sr = row.get("audio.sampling_rate", 16000)
        else:
            _LOGGER.warning("Row %s: no 'audio' column, skipping", idx)
            return None

    if isinstance(audio_col, dict) and "array" in audio_col:
        arr = np.asarray(audio_col["array"], dtype=np.float32)
        sr = audio_col.get("sampling_rate", 16000)
        return _normalize_audio(arr, sr)

    if isinstance(audio_col, dict) and "bytes" in audio_col:
        raw_bytes = audio_col["bytes"]
        result = _decode_audio_bytes(raw_bytes)
        if result is not None:
            arr, actual_sr = result
            return _normalize_audio(arr, actual_sr)

    if isinstance(audio_col, bytes):
        result = _decode_audio_bytes(audio_col)
        if result is not None:
            arr, actual_sr = result
            return _normalize_audio(arr, actual_sr)

    if isinstance(audio_col, (list, np.ndarray)):
        arr = np.asarray(audio_col, dtype=np.float32)
        return _normalize_audio(arr, sr)

    _LOGGER.warning("Row %s: unsupported audio column type %s, keys=%s", idx,
                    type(audio_col),
                    list(audio_col) if isinstance(audio_col, dict) else "N/A")
    return None


def _normalize_audio(arr: np.ndarray, sr: int) -> np.ndarray:
    """Ensure mono, float32, 16kHz, and in [-1, 1] range.

    Args:
        arr: Audio array, any shape and dtype.
        sr: Original sample rate.

    Returns:
        1-D float32 array at 16 kHz, normalized to [-1, 1].
    """
    arr = arr.astype(np.float32)

    # Flatten if multi-dimensional
    if arr.ndim > 1:
        arr = arr.reshape(-1)

    # Auto-scale integer audio to [-1, 1]
    if np.abs(arr).max() > 2.0:
        arr = arr / 32768.0

    # Resample if needed
    if sr != 16000:
        import scipy.signal
        desired_len = int(round(len(arr) / sr * 16000))
        arr = scipy.signal.resample(arr, desired_len).astype(np.float32)

    return arr


def _decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int] | None:
    """Decode a recording or a bare 16-bit PCM buffer.

    Args:
        raw_bytes: Encoded file contents or little-endian signed PCM at 16 kHz.

    Returns:
        The waveform and its sample rate, or None for an empty buffer.

    Raises:
        ValueError: If the PCM buffer does not contain complete 16-bit samples.
    """
    try:
        return whisper_preprocess.read_audio(io.BytesIO(raw_bytes))
    except (ValueError, RuntimeError):
        # Some datasets store bare 16 kHz, signed 16-bit PCM in this field.
        if raw_bytes.startswith((b"RIFF", b"fLaC", b"OggS", b"ID3")):
            raise
        pcm = np.frombuffer(raw_bytes, dtype="<i2")
        if pcm.size:
            return pcm.astype(np.float32) / 32768.0, 16000
        return None


def process_audio(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Compute the normalized log-Mel spectrogram for one audio sample.

    Args:
        audio: 1-D float32 audio array.
        mel_filters: Precomputed Mel filterbank matrix.

    Returns:
        Float32 array of shape ``(1, 80, 3000, 1)`` — 4D NHWC format
        matching the ONNX model input expected by hb_mapper.
    """
    return whisper_preprocess.encoder_input(audio, mel_filters)


def main() -> None:
    """Generate raw float32 encoder inputs from recordings or a Parquet file."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Generate Whisper encoder calibration data",
        add_help=True,
    )

    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory containing WAV audio files.",
    )
    input_group.add_argument(
        "--parquet-file",
        type=str,
        default=None,
        help="Path to a Parquet dataset file (HuggingFace datasets format).",
    )

    parser.add_argument(
        "--mel-filters",
        type=str,
        default=os.path.join(_SAMPLE_DIR, "model", "mel_80_filters.txt"),
        help="Path to the Mel filterbank data file (mel_80_filters.txt).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save headerless float32 .bin calibration files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Maximum number of calibration samples to generate (default: 50).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.mel_filters):
        _LOGGER.error("Mel filterbank file not found: %s", args.mel_filters)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load mel filterbank (shared across all samples)
    mel_filters = whisper_preprocess.load_mel_filters(args.mel_filters)

    if args.parquet_file:
        if not os.path.isfile(args.parquet_file):
            _LOGGER.error("Parquet file not found: %s", args.parquet_file)
            sys.exit(1)
        _LOGGER.info("Reading parquet: %s", args.parquet_file)
        samples = read_parquet_audio(args.parquet_file, args.num_samples)
    else:
        if not os.path.isdir(args.audio_dir):
            _LOGGER.error("Audio directory not found: %s", args.audio_dir)
            sys.exit(1)
        audio_files = collect_audio_files(args.audio_dir, args.num_samples)
        if not audio_files:
            _LOGGER.error("No audio files found in %s", args.audio_dir)
            sys.exit(1)
        # Load all into memory for uniform processing
        samples = []
        for audio_path in audio_files:
            name = os.path.splitext(os.path.basename(audio_path))[0]
            samples.append((name, whisper_preprocess.load_audio(audio_path)))

    total = len(samples)
    if not samples:
        _LOGGER.error("No audio samples found")
        sys.exit(1)
    _LOGGER.info("Processing %d samples for calibration", total)

    success = 0
    for idx, (name, audio) in enumerate(samples, start=1):
        out_path = os.path.join(args.output_dir, f"{name}.bin")

        try:
            mel = process_audio(audio, mel_filters)
            mel.tofile(out_path)
            _LOGGER.info("[%d/%d] Saved %s  shape=%s", idx, total, out_path,
                         mel.shape)
            success += 1
        except Exception as error:
            _LOGGER.warning("[%d/%d] Failed %s: %s", idx, total, name, error)

    _LOGGER.info("Done. Generated %d/%d calibration samples in %s", success,
                 total, args.output_dir)

    if success == 0:
        sys.exit("No calibration tensors were generated")
    if success < 20:
        _LOGGER.warning("Fewer than 20 calibration samples generated. "
                        "INT8 calibration quality may be suboptimal. "
                        "Consider providing more audio data.")


if __name__ == "__main__":
    main()
