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
"""Transcribe an audio recording with Whisper on the RDK X5."""

from __future__ import annotations

import argparse
import logging
import os

import whisper

_LOGGER = logging.getLogger("Whisper")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_ENCODER_MODEL_PATH = whisper.WhisperConfig.encoder_model_path
DEFAULT_DECODER_MODEL_PATH = whisper.WhisperConfig.decoder_model_path
LEGACY_DECODER_PATH = os.path.join(MODEL_DIR,
                                   "whisper_decoder_tiny_80x3000_nv12.bin")
if not os.path.isfile(DEFAULT_DECODER_MODEL_PATH) and os.path.isfile(
        LEGACY_DECODER_PATH):
    DEFAULT_DECODER_MODEL_PATH = LEGACY_DECODER_PATH
DEFAULT_VOCAB_PATH = whisper.WhisperConfig.vocab_path
DEFAULT_MEL_FILTERS_PATH = whisper.WhisperConfig.mel_filters_path
DEFAULT_TEST_AUDIO = os.path.join(TEST_DATA_DIR, "test_audio.wav")


def save_text(path: str, text: str) -> None:
    """Save the transcription result to a text file.

    Args:
        path: Path to the output text file.
        text: Transcribed text to save.
    """

    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        output_file.write(text)
    _LOGGER.info("Transcription saved to %s", path)


def main() -> None:
    """Transcribe the input audio and write the result to a text file."""

    logging.basicConfig(
        level=logging.INFO,
        format=("[%(name)s] [%(asctime)s.%(msecs)03d] "
                "[%(levelname)s] %(message)s"),
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Whisper ASR Inference")
    parser.add_argument("--encoder-model-path",
                        type=str,
                        default=DEFAULT_ENCODER_MODEL_PATH,
                        help="Path to the BPU encoder *.bin model.")
    parser.add_argument(
        "--decoder-model-path",
        type=str,
        default=DEFAULT_DECODER_MODEL_PATH,
        help="Path to the CPU ONNX decoder or a precompiled .bin decoder.")
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Whisper .tiktoken file or legacy vocabulary text file.")
    parser.add_argument("--mel-filters-path",
                        type=str,
                        default=DEFAULT_MEL_FILTERS_PATH,
                        help="Path to the Mel filterbank data file.")
    parser.add_argument("--priority",
                        type=int,
                        default=0,
                        help="Model priority (0~255).")
    parser.add_argument("--bpu-cores",
                        nargs="+",
                        type=int,
                        default=[0],
                        help="BPU core indexes to run inference.")
    parser.add_argument("--test-audio",
                        type=str,
                        default=DEFAULT_TEST_AUDIO,
                        help="Path to the test input audio file.")
    parser.add_argument("--text-save-path",
                        type=str,
                        default=os.path.join(TEST_DATA_DIR, "result.txt"),
                        help="Path to save the transcribed text.")
    parser.add_argument("--language",
                        type=str,
                        choices=("en", "zh"),
                        default="en",
                        help="Language code for transcription (e.g., en, zh).")
    args = parser.parse_args()

    vocab_path = args.vocab_path or DEFAULT_VOCAB_PATH
    if args.vocab_path is None and not os.path.isfile(vocab_path):
        vocab_path = os.path.join(MODEL_DIR, f"vocab_{args.language}.txt")
    config = whisper.WhisperConfig(
        encoder_model_path=args.encoder_model_path,
        decoder_model_path=args.decoder_model_path,
        vocab_path=vocab_path,
        mel_filters_path=args.mel_filters_path,
        language=args.language,
    )
    model = whisper.Whisper(config)
    model.set_scheduling_params(priority=args.priority,
                                bpu_cores=args.bpu_cores)

    _LOGGER.info("Loaded %s with inputs %s", model.encoder.model_names,
                 model.encoder.input_shapes)

    audio = whisper.load_audio(args.test_audio)
    text = model.predict(audio)

    _LOGGER.info("Transcription result: %s", text)
    save_text(args.text_save_path, text)


if __name__ == "__main__":
    main()
