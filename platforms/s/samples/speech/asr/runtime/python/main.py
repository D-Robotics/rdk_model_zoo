# Copyright (c) 2025 D-Robotics Corporation
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

"""ASR inference entry script.

This script runs a BPU-quantized ASR (.hbm) model on a single audio file
and prints the transcribed text.

Workflow:
    1) Parse CLI arguments.
    2) Check platform compatibility (S100 only).
    3) Download the model file if missing.
    4) Load vocabulary JSON -> build id2token mapping.
    5) Create ASRConfig and initialize ASR runtime wrapper.
    6) Stream audio in fixed-length chunks: preprocess -> inference -> decode.
    7) Print full transcription.

Notes:
    - This model only supports RDK S100 platform.
    - If running on RDK S600, inference will not produce correct results.
      Please refer to README.md for platform compatibility details.
    - The project root is appended to sys.path to import shared utilities
      under `utils/py_utils/`.

Example:
    python main.py \\
        --audio-file ../../test_data/chi_sound.wav \\
        --vocab-file ../../test_data/vocab.json
"""

import os
import sys
import json
import argparse

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
from asr import ASR, ASRConfig


def main() -> None:
    """Run ASR speech recognition on an audio file.

    This function parses command-line arguments, loads the ASR model and
    vocabulary, and runs streaming inference to produce a full transcription.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()
    model_download_url = (f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                          f"rdk_{soc}/asr/asr.hbm")

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/asr.hbm',
                        help='Path to BPU quantized *.hbm model file.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--audio-file', type=str, default='../../test_data/chi_sound.wav',
                        help='Path to input audio file (.wav or .flac).')
    parser.add_argument('--vocab-file', type=str, default='../../test_data/vocab.json',
                        help='Path to vocabulary JSON file (token -> id mapping).')
    parser.add_argument('--audio-maxlen', type=int, default=30000,
                        help='Number of audio samples per inference chunk at new_rate Hz.')
    parser.add_argument('--new-rate', type=int, default=16000,
                        help='Target audio sample rate in Hz.')

    opt = parser.parse_args()

    # Download model if missing
    file_io.download_model_if_needed(opt.model_path, model_download_url)

    # Load vocabulary: JSON is {"token": id}, build id->token list
    with open(opt.vocab_file, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}

    # Initialize ASR configuration and model
    config = ASRConfig(
        model_path=opt.model_path,
        audio_maxlen=opt.audio_maxlen,
        new_rate=opt.new_rate,
    )
    model = ASR(config)

    # Configure runtime scheduling (BPU cores, priority)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(model.model)

    # Run full pipeline: streaming chunks -> concatenated transcription
    full_text = model.predict(opt.audio_file, id2token)

    print("Transcription:")
    print(full_text)


if __name__ == "__main__":
    main()
