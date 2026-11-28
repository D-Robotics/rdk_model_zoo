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
"""Transcribe an audio recording with Wav2Vec2 on the RDK X5."""

from __future__ import annotations

import argparse
import logging
import os

import wav2vec2

_LOGGER = logging.getLogger("Wav2Vec2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = wav2vec2.Wav2Vec2Config.model_path
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

    parser = argparse.ArgumentParser(description="Wav2Vec2 ASR Inference")
    parser.add_argument("--model-path",
                        type=str,
                        default=DEFAULT_MODEL_PATH,
                        help="Path to the BPU quantized *.bin model.")
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
                        default="en",
                        help="Language code for transcription (e.g., en, zh).")
    parser.add_argument("--blank-id",
                        type=int,
                        default=0,
                        help="CTC blank token ID.")
    parser.add_argument("--vocab-path",
                        type=str,
                        default=None,
                        help="CTC token-to-ID JSON for this checkpoint.")
    args = parser.parse_args()

    config = wav2vec2.Wav2Vec2Config(
        model_path=args.model_path,
        language=args.language,
        blank_id=args.blank_id,
        vocab_path=args.vocab_path,
    )
    model = wav2vec2.Wav2Vec2(config)
    model.set_scheduling_params(priority=args.priority,
                                bpu_cores=args.bpu_cores)

    _LOGGER.info("Loaded %s with inputs %s", model.model.model_names,
                 model.model.input_shapes)

    audio = wav2vec2.load_audio(args.test_audio, num_samples=model.num_samples)
    text = model.predict(audio)

    _LOGGER.info("Transcription result: %s", text)
    save_text(args.text_save_path, text)


if __name__ == "__main__":
    main()
