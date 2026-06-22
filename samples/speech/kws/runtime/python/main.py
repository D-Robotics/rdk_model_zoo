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

"""KWS inference entry script.

This script runs a BPU-quantized KWS (.hbm) model on a single audio file
and prints the keyword detection confidence score.

Workflow:
    1) Parse CLI arguments.
    2) Check platform compatibility (S100 only).
    3) Download the model file if missing.
    4) Create KWSConfig and initialize KWS runtime wrapper.
    5) Run inference: preprocess -> forward -> postprocess.
    6) Print confidence score.

Notes:
    - This model only supports RDK S100 platform.
    - If running on RDK S600, inference will not produce correct results.
      Please refer to README.md for platform compatibility details.
    - The project root is appended to sys.path to import shared utilities
      under `utils/py_utils/`.

Example:
    python main.py \\
        --audio-file ../../test_data/sample.wav
"""

import os
import sys
import argparse

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
from kws import KWS, KWSConfig


def main() -> None:
    """Run KWS keyword spotting on an audio file.

    This function parses command-line arguments, loads the KWS model,
    and runs inference to produce a confidence score.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()
    model_download_url = (f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
                          f"rdk_{soc}/kws/kws.hbm")

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/kws.hbm',
                        help='Path to BPU quantized *.hbm model file.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--audio-file', type=str, default='../../test_data/sample.wav',
                        help='Path to input audio file (.wav).')
    parser.add_argument('--audio-maxlen', type=int, default=60000,
                        help='Maximum number of audio samples before truncation.')
    parser.add_argument('--frame-shift', type=int, default=10,
                        help='Frame shift in milliseconds for fbank extraction.')
    parser.add_argument('--frame-length', type=int, default=25,
                        help='Frame length in milliseconds for fbank extraction.')
    parser.add_argument('--n-mels', type=int, default=80,
                        help='Number of mel filter banks for fbank extraction.')

    opt = parser.parse_args()

    # Download model if missing
    file_io.download_model_if_needed(opt.model_path, model_download_url)

    # Initialize KWS configuration and model
    config = KWSConfig(
        model_path=opt.model_path,
        audio_maxlen=opt.audio_maxlen,
        frame_shift=opt.frame_shift,
        frame_length=opt.frame_length,
        n_mels=opt.n_mels,
    )
    model = KWS(config)

    # Configure runtime scheduling (BPU cores, priority)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(model.model)

    # Run full pipeline
    score = model.predict(opt.audio_file)

    print(f"Keyword confidence score: {score:.4f}")


if __name__ == "__main__":
    main()
