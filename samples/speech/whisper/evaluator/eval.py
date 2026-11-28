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
"""Whisper ASR Evaluation Script.

Evaluates the Whisper model on a test dataset of audio files with
corresponding ground truth transcriptions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time

# Prefer the sample runtime when a package with the same name is installed.
_SAMPLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST_DATA_DIR = os.path.join(_SAMPLE_DIR, "test_data")
sys.path.insert(0, os.path.join(_SAMPLE_DIR, "runtime", "python"))

import whisper

_LOGGER = logging.getLogger("Whisper_Eval")


def load_ground_truth(json_path: str) -> dict[str, str]:
    """Load ground truth transcriptions from a JSON file.

    Expects a JSON file mapping audio file names to reference texts:
        {"audio1.wav": "hello world", "audio2.wav": "how are you"}

    Args:
        json_path: Path to the ground truth JSON file.

    Returns:
        A dict mapping audio file names to reference texts.
    """
    with open(json_path, encoding="utf-8") as reference_file:
        references = json.load(reference_file)
    _LOGGER.info("Loaded %d ground truth entries from %s", len(references),
                 json_path)
    return references


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.

    Args:
        reference: Ground truth text string.
        hypothesis: Predicted text string.

    Returns:
        WER as a float (0.0 = perfect match).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    # Levenshtein distance at word level
    distances = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        distances[i][0] = i
    for j in range(len(hyp_words) + 1):
        distances[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = min(distances[i - 1][j], distances[i][j - 1],
                                      distances[i - 1][j - 1]) + 1

    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return distances[len(ref_words)][len(hyp_words)] / len(ref_words)


def main() -> None:
    """Evaluate matching recordings and save per-file word error rates."""

    logging.basicConfig(
        level=logging.INFO,
        format=("[%(name)s] [%(asctime)s.%(msecs)03d] "
                "[%(levelname)s] %(message)s"),
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Whisper ASR Evaluation")
    parser.add_argument("--encoder-model-path",
                        type=str,
                        default=whisper.WhisperConfig.encoder_model_path,
                        help="Path to the BPU encoder *.bin model.")
    parser.add_argument(
        "--decoder-model-path",
        type=str,
        default=whisper.WhisperConfig.decoder_model_path,
        help="Path to the CPU ONNX decoder or a precompiled .bin decoder.")
    parser.add_argument("--vocab-path",
                        type=str,
                        default=whisper.WhisperConfig.vocab_path,
                        help="Path to the vocabulary text file.")
    parser.add_argument("--mel-filters-path",
                        type=str,
                        default=whisper.WhisperConfig.mel_filters_path,
                        help="Path to the Mel filterbank data file.")
    parser.add_argument("--audio-dir",
                        type=str,
                        default=_TEST_DATA_DIR,
                        help="Directory containing test audio files.")
    parser.add_argument("--ground-truth",
                        type=str,
                        default=os.path.join(_TEST_DATA_DIR, "reference.json"),
                        help="Path to the ground truth JSON file.")
    parser.add_argument("--json-save-path",
                        type=str,
                        default="whisper_asr_results.json",
                        help="Path to save evaluation results.")
    parser.add_argument("--language",
                        type=str,
                        default="en",
                        help="Language code (e.g., en, zh).")
    parser.add_argument("--priority",
                        type=int,
                        default=0,
                        help="Model scheduling priority (0~255).")
    parser.add_argument("--bpu-cores",
                        nargs="+",
                        type=int,
                        default=[0],
                        help="BPU core indices.")
    args = parser.parse_args()

    for path in (args.encoder_model_path, args.decoder_model_path,
                 args.vocab_path, args.mel_filters_path):
        if not os.path.isfile(path):
            parser.error(f"model or resource file not found: {path}")
    if not os.path.exists(args.ground_truth):
        parser.error(f"ground truth file not found: {args.ground_truth}")
    if not os.path.isdir(args.audio_dir):
        parser.error(f"audio directory not found: {args.audio_dir}")

    references = load_ground_truth(args.ground_truth)

    config = whisper.WhisperConfig(
        encoder_model_path=args.encoder_model_path,
        decoder_model_path=args.decoder_model_path,
        vocab_path=args.vocab_path,
        mel_filters_path=args.mel_filters_path,
        language=args.language,
    )
    model = whisper.Whisper(config)
    model.set_scheduling_params(priority=args.priority,
                                bpu_cores=args.bpu_cores)

    audio_extensions = (".wav", ".mp3", ".flac")
    audio_files = []
    for filename in sorted(os.listdir(args.audio_dir)):
        if filename.lower().endswith(audio_extensions):
            audio_files.append(filename)

    total_files = len(audio_files)
    _LOGGER.info("Found %d audio files, %d ground truth entries", total_files,
                 len(references))

    total_wer = 0.0
    matched = 0
    t_start = time.perf_counter()
    results = []

    for idx, audio_name in enumerate(audio_files):
        audio_path = os.path.join(args.audio_dir, audio_name)

        ref_text = references.get(audio_name)
        if ref_text is None:
            _LOGGER.debug("No ground truth for %s, skipping", audio_name)
            continue

        # Each backend reports failures differently. Continue with other files.
        try:
            audio = whisper.load_audio(audio_path)
            text = model.predict(audio)
        except Exception as error:
            _LOGGER.error("Error processing %s: %s", audio_name, error)
            continue

        matched += 1
        wer = compute_wer(ref_text, text)
        total_wer += wer
        results.append({
            "file": audio_name,
            "reference": ref_text,
            "hypothesis": text,
            "wer": wer,
        })

        if (idx + 1) % 10 == 0:
            _LOGGER.info("Processed %d/%d files", idx + 1, total_files)

    elapsed = time.perf_counter() - t_start
    if not matched:
        parser.error("no audio files with reference transcripts "
                     "were successfully evaluated")
    avg_wer = total_wer / matched

    summary = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.encoder_model_path,
        "decoder_model": args.decoder_model_path,
        "total_files": total_files,
        "matched": matched,
        "avg_wer": round(avg_wer, 4),
        "elapsed_sec": round(elapsed, 2),
        "results": results,
    }

    with open(args.json_save_path, "w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=False)

    _LOGGER.info("Evaluation complete: avg WER=%.4f, matched=%d/%d", avg_wer,
                 matched, total_files)
    _LOGGER.info("Results saved to %s", args.json_save_path)


if __name__ == "__main__":
    main()
