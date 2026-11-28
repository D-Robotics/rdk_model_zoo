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
"""Prepare fixed-length float32 waveforms for Wav2Vec2 X5 calibration."""

import argparse
import math
from pathlib import Path
import sys

import onnx

_SAMPLE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(_SAMPLE_DIR.parents[2] / "utils/py_utils"))
import audio_utils


def main() -> None:
    """Write raw calibration tensors in the waveform shape exported by ONNX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir",
                        required=True,
                        type=Path,
                        help="Directory of source recordings.")
    parser.add_argument("--model",
                        type=Path,
                        default=_SAMPLE_DIR / "model/wav2vec2_base.onnx",
                        help="ONNX model defining the input shape.")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=_SAMPLE_DIR / "cal_data",
                        help="Directory for raw float32 tensors.")
    parser.add_argument("--num-samples",
                        type=int,
                        default=50,
                        help="Maximum number of recordings to use.")
    args = parser.parse_args()
    graph = onnx.load(str(args.model), load_external_data=False).graph
    weights = {tensor.name for tensor in graph.initializer}
    inputs = [value for value in graph.input if value.name not in weights]
    if len(inputs) != 1:
        parser.error("expected a single waveform input")
    shape = [dim.dim_value for dim in inputs[0].type.tensor_type.shape.dim]
    if not shape or any(n <= 0 for n in shape) or shape[0] != 1:
        parser.error(f"expected a static batch-one waveform, got {shape}")
    files = sorted(
        path for path in args.audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".wav", ".flac", ".mp3"))
    if not files or args.num_samples <= 0:
        parser.error("provide audio files and a positive --num-samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = math.prod(shape)
    for path in files[:args.num_samples]:
        audio = audio_utils.load_waveform(path)
        waveform = audio_utils.wav2vec2_input(audio, num_samples)
        waveform.tofile(args.output_dir / (path.name + ".bin"))
    count = min(len(files), args.num_samples)
    print(f"Wrote {count} float32 tensors of shape {shape}")


if __name__ == "__main__":
    main()
