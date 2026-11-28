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
"""Export a static X5 Whisper encoder and a variable-length CPU decoder."""

import argparse
import contextlib
import inspect
from pathlib import Path
import shutil

import onnx
import onnxsim
import torch
import whisper


class EncoderWrapper(torch.nn.Module):
    """Adapt a Whisper encoder to the X5 featuremap input layout.

    Attributes:
        encoder: The checkpoint's PyTorch audio encoder.
    """

    def __init__(self, encoder):
        """Wrap the audio encoder for export.

        Args:
            encoder: Whisper audio encoder from the loaded checkpoint.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        """Encode log-Mel features after dropping the singleton channel.

        Args:
            mel: Input of shape (batch, Mel bins, frames, 1).

        Returns:
            Hidden states for the decoder.
        """
        return self.encoder(mel.squeeze(-1))


def export(module, inputs, output, names, dynamic_axes=None):
    """Export and simplify one Whisper ONNX graph.

    Args:
        module: PyTorch encoder or decoder in evaluation mode.
        inputs: Example input tensors in the model's positional order.
        output: Destination ONNX file.
        names: Names of the graph inputs in positional order.
        dynamic_axes: Optional symbolic names for axes of inputs and outputs.

    Raises:
        ValueError: If the simplified graph fails validation.
    """
    options = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    torch.onnx.export(module,
                      inputs,
                      str(output),
                      input_names=names,
                      output_names=["out"],
                      opset_version=11,
                      dynamic_axes=dynamic_axes,
                      **options)
    simplified, valid = onnxsim.simplify(onnx.load(str(output)))
    if not valid:
        raise ValueError(f"ONNX simplification failed validation for {output}")
    onnx.save(simplified, str(output))
    print(output, flush=True)


def main() -> None:
    """Export the encoder, decoder and vocabulary for a Whisper checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type",
                        "--model_type",
                        type=str,
                        default="tiny",
                        help="Whisper variant used to name the export")
    parser.add_argument("--checkpoint",
                        type=Path,
                        default=None,
                        help="Local .pt file for offline export")
    parser.add_argument(
        "--n-mels",
        "--n_mels",
        type=int,
        default=None,
        help="Expected number of Mel channels in this checkpoint")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path(__file__).resolve().parents[2] / "model",
                        help="Directory for the ONNX exports and vocabulary.")
    args = parser.parse_args()
    model = whisper.load_model(str(args.checkpoint or args.model_type),
                               device="cpu").eval()
    dims = model.dims
    if args.n_mels is not None and args.n_mels != dims.n_mels:
        parser.error(f"checkpoint expects {dims.n_mels} Mel channels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mel = torch.zeros(1, dims.n_mels, 2 * dims.n_audio_ctx, 1)
    audio = torch.zeros(1, dims.n_audio_ctx, dims.n_audio_state)
    tokens = torch.arange(4, dtype=torch.long).unsqueeze(0)
    eager_attention = getattr(whisper.model, "disable_sdpa",
                              contextlib.nullcontext)
    encoder_path = args.output_dir / f"whisper_encoder_{args.model_type}.onnx"
    decoder_path = args.output_dir / f"whisper_decoder_{args.model_type}.onnx"
    text_axes = {"tokens": {1: "text_length"}, "out": {1: "text_length"}}
    with torch.no_grad(), eager_attention():
        export(EncoderWrapper(model.encoder), (mel,), encoder_path, ["x"])
        export(model.decoder, (tokens, audio),
               decoder_path, ["tokens", "audio"],
               dynamic_axes=text_axes)
    vocab = "gpt2.tiktoken"
    if model.is_multilingual:
        vocab = "multilingual.tiktoken"
    shutil.copyfile(
        Path(whisper.__file__).parent / "assets" / vocab,
        args.output_dir / vocab)


if __name__ == "__main__":
    main()
