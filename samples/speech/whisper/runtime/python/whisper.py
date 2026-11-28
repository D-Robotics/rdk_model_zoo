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
"""Whisper inference with an X5 encoder and an autoregressive decoder."""

from __future__ import annotations

import base64
import dataclasses
from pathlib import Path
import time

import numpy as np

import whisper_preprocess as preprocess

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
load_audio = preprocess.load_audio


@dataclasses.dataclass
class WhisperConfig:
    """Model and resource paths for a Whisper transcription.

    Attributes:
        encoder_model_path: Compiled X5 encoder.
        decoder_model_path: CPU ONNX decoder or compiled board decoder.
        vocab_path: Tiktoken vocabulary or legacy text vocabulary.
        mel_filters_path: Text file containing the 80-channel Mel filterbank.
        language: Transcription language, either English (en) or Chinese (zh).
    """

    encoder_model_path: str = str(MODEL_DIR /
                                  "whisper_encoder_tiny_80x3000_x5.bin")
    decoder_model_path: str = str(MODEL_DIR / "whisper_decoder_tiny.onnx")
    vocab_path: str = str(MODEL_DIR / "multilingual.tiktoken")
    mel_filters_path: str = str(MODEL_DIR / "mel_80_filters.txt")
    language: str = "en"


class WhisperDecoder:
    """Greedy decoder for multilingual Whisper encoder states.

    Attributes:
        board_runtime: Loaded board decoder, or None for a CPU ONNX decoder.
    """

    def __init__(self, model_path: str, vocab_path: str, language: str):
        """Load a decoder and the vocabulary used by its checkpoint.

        Args:
            model_path: ONNX or compiled decoder file.
            vocab_path: Tiktoken or legacy text vocabulary file.
            language: Language code used to construct the decoder prompt.
        """
        self.language_token = {
            "en": preprocess.TASK_EN,
            "zh": preprocess.TASK_ZH
        }[language]
        self.board_runtime = None
        if Path(model_path).suffix.lower() == ".onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"])
            token_input, audio_input = self.session.get_inputs()
            self.token_name = token_input.name
            self.audio_name = audio_input.name
            self.token_shape = token_input.shape
            self.audio_shape = audio_input.shape
        else:
            import hbm_runtime

            self.board_runtime = hbm_runtime.HB_HBMRuntime(str(model_path))
            self.model_name = self.board_runtime.model_names[0]
            input_names = self.board_runtime.input_names[self.model_name]
            self.token_name, self.audio_name = input_names
            shapes = self.board_runtime.input_shapes[self.model_name]
            self.token_shape = shapes[self.token_name]
            self.audio_shape = shapes[self.audio_name]
            self.output_name = self.board_runtime.output_names[
                self.model_name][0]

        self.token_bytes = None
        if Path(vocab_path).suffix == ".tiktoken":
            self.token_bytes = {}
            with open(vocab_path, "rb") as vocab_file:
                for line in vocab_file:
                    if line.strip():
                        piece, rank = line.split()
                        self.token_bytes[int(rank)] = base64.b64decode(piece)
        else:
            self.vocab = preprocess.read_vocab(vocab_path)

    def decode(self, encoder_output: np.ndarray) -> str:
        """Generate one transcription from the encoder hidden states.

        Args:
            encoder_output: Hidden states for a single audio segment.

        Returns:
            Decoded text with surrounding whitespace removed.

        Raises:
            ValueError: If a fixed-length decoder cannot fit the prompt.
        """
        audio = np.asarray(encoder_output, dtype=np.float32)
        audio = audio.reshape(self.audio_shape)
        prefix = [
            preprocess.SOT, self.language_token, preprocess.TRANSCRIBE,
            preprocess.NO_TIMESTAMPS
        ]
        fixed_length = None
        if all(
                isinstance(dim, (int, np.integer)) and dim > 0
                for dim in self.token_shape):
            fixed_length = int(np.prod(self.token_shape))
            if fixed_length < len(prefix):
                raise ValueError("decoder token input is shorter than the "
                                 "Whisper prompt")

        # Whisper Tiny normally generates up to half its 448-token context.
        generated = []
        for _ in range(224):
            tokens = prefix + generated
            position = len(tokens) - 1
            ids = np.asarray([tokens], dtype=np.int64)
            if fixed_length is not None:
                padding = ((0, 0), (0, fixed_length - len(tokens)))
                ids = np.pad(ids, padding, constant_values=preprocess.END_TOKEN)
                ids = ids.reshape(self.token_shape)
            inputs = {self.token_name: ids, self.audio_name: audio}
            if self.board_runtime is None:
                logits = self.session.run(None, inputs)[0]
            else:
                outputs = self.board_runtime.run({self.model_name: inputs})
                logits = outputs[self.model_name][self.output_name]
            scores = logits.reshape(-1, logits.shape[-1])[position]
            token = int(np.argmax(scores[:preprocess.END_TOKEN + 1]))
            if token == preprocess.END_TOKEN:
                break
            generated.append(token)
            if fixed_length is not None and len(tokens) == fixed_length:
                break

        if self.token_bytes is not None:
            pieces = (self.token_bytes[token] for token in generated)
            return b"".join(pieces).decode("utf-8", errors="replace").strip()
        pieces = (self.vocab.get(str(token), "") for token in generated)
        text = "".join(pieces).replace("\u0120", " ")
        if self.language_token == preprocess.TASK_ZH:
            text = preprocess.base64_decode(text)
        return text.strip()


class Whisper:
    """Transcribe 16 kHz audio with a Whisper encoder and decoder.

    Attributes:
        cfg: Configuration used to load the models and resources.
        encoder: X5 runtime exposing the encoder's input and output metadata.
        decoder: Greedy decoder using the configured language and vocabulary.
    """

    def __init__(self, config: WhisperConfig):
        """Load both models and the filterbank required by the encoder.

        Args:
            config: Paths to the model files, vocabulary and Mel filterbank.
        """
        import hbm_runtime

        self.cfg = config
        self.encoder = hbm_runtime.HB_HBMRuntime(config.encoder_model_path)
        self.enc_model_name = self.encoder.model_names[0]
        self.enc_input_names = self.encoder.input_names[self.enc_model_name]
        self.enc_output_names = self.encoder.output_names[self.enc_model_name]
        self.decoder = WhisperDecoder(config.decoder_model_path,
                                      config.vocab_path, config.language)
        self.mel_filters = preprocess.load_mel_filters(config.mel_filters_path)

    def set_scheduling_params(self,
                              priority: int | None = None,
                              bpu_cores: list[int] | None = None) -> None:
        """Apply scheduling options to each loaded board model.

        Args:
            priority: Optional scheduling priority from 0 to 255.
            bpu_cores: Optional list of BPU core indices.
        """
        for runtime in (self.encoder, self.decoder.board_runtime):
            if runtime is None:
                continue
            name = runtime.model_names[0]
            params = {}
            if priority is not None:
                params["priority"] = {name: priority}
            if bpu_cores is not None:
                params["bpu_cores"] = {name: bpu_cores}
            if params:
                runtime.set_scheduling_params(**params)

    def pre_process(self,
                    audio: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """Build the log-Mel input for the encoder.

        Args:
            audio: Mono 16 kHz waveform.

        Returns:
            The (1, 80, 3000, 1) input keyed by model and tensor name.
        """
        mel = preprocess.encoder_input(audio, self.mel_filters)
        inputs = {self.enc_input_names[0]: mel}
        return {self.enc_model_name: inputs}

    def forward(
        self, inputs: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run the encoder on the X5.

        Args:
            inputs: Nested input dictionary returned by pre_process().

        Returns:
            The runtime's output dictionary keyed by model and tensor name.
        """
        return self.encoder.run(inputs)

    def post_process(self, encoder_outputs: dict) -> str:
        """Decode the encoder output into a transcription.

        Args:
            encoder_outputs: Raw runtime output or the selected model's tensors.

        Returns:
            The decoded transcript.
        """
        tensors = encoder_outputs.get(self.enc_model_name, encoder_outputs)
        return self.decoder.decode(tensors[self.enc_output_names[0]])

    def predict(self, audio: np.ndarray) -> str:
        """Run preprocessing, inference and decoding for one audio segment.

        Args:
            audio: Mono 16 kHz waveform, padded or trimmed to 30 seconds.

        Returns:
            The decoded transcript.
        """
        start = time.perf_counter()
        inputs = self.pre_process(audio)
        preprocessed = time.perf_counter()
        encoder_outputs = self.forward(inputs)
        encoded = time.perf_counter()
        text = self.post_process(encoder_outputs)
        decoded = time.perf_counter()

        print(f"\n[Log] Pre-process: {(preprocessed - start) * 1000:.2f} ms | "
              f"Encoder: {(encoded - preprocessed) * 1000:.2f} ms | "
              f"Decoder: {(decoded - encoded) * 1000:.2f} ms")
        return text

    def __call__(self, audio: np.ndarray) -> str:
        """Transcribe one audio segment.

        Args:
            audio: Mono 16 kHz waveform.

        Returns:
            The same transcript as predict().
        """
        return self.predict(audio)
