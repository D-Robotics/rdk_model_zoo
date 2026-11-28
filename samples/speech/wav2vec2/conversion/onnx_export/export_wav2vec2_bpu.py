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
"""Export fixed-length Wav2Vec2 CTC logits for X5 featuremap compilation."""

import argparse
import inspect
from pathlib import Path

import torch
import transformers


class CTCLogits(torch.nn.Module):
    """Expose a Wav2Vec2 checkpoint's CTC logits as one ONNX output.

    Attributes:
        model: The PyTorch CTC model being exported.
    """

    def __init__(self, model):
        """Wrap a trained CTC model.

        Args:
            model: Wav2Vec2 model with a CTC projection head.
        """
        super().__init__()
        self.model = model

    def forward(self, waveform):
        """Compute frame-level CTC logits.

        Args:
            waveform: Batch of fixed-length mono waveforms.

        Returns:
            Logits indexed by batch, time frame and token.
        """
        return self.model(waveform, return_dict=False)[0]


def main() -> None:
    """Export a fixed-length waveform model using ONNX opset 11."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="facebook/wav2vec2-base-960h",
        help="Local pretrained directory or Hugging Face model ID")
    parser.add_argument("--samples",
                        type=int,
                        default=65535,
                        help="Fixed input length in 16 kHz audio samples")
    parser.add_argument("--output",
                        type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        "model/wav2vec2_base.onnx",
                        help="Destination ONNX model file.")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    config = transformers.Wav2Vec2Config.from_pretrained(
        args.checkpoint, attn_implementation="eager")
    model = transformers.Wav2Vec2ForCTC.from_pretrained(args.checkpoint,
                                                        config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Older PyTorch already uses the legacy opset-11 exporter.
    options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(
            CTCLogits(model).eval(),
            torch.zeros(1, args.samples),
            str(args.output),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            **options,
        )
    print(f"{args.output}: {args.checkpoint}, input [1, {args.samples}]")


if __name__ == "__main__":
    main()
