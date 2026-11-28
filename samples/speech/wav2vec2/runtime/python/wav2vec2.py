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
"""Wav2Vec2 inference and greedy CTC decoding for RDK X5."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

# Direct sample execution uses the same audio preparation as calibration.
sys.path.append(str(Path(__file__).resolve().parents[5] / "utils/py_utils"))
import audio_utils

SAMPLE_RATE = 16000
MAX_AUDIO_SAMPLES = SAMPLE_RATE * 15  # 15 seconds max
MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


def load_audio(audio_path: str,
               target_sr: int = SAMPLE_RATE,
               num_samples: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    """Load a fixed-length mono waveform.

    Args:
        audio_path: Audio file to decode.
        target_sr: Output sample rate in hertz.
        num_samples: Required waveform length in samples.

    Returns:
        A one-dimensional float32 waveform.
    """
    return audio_utils.load_waveform(audio_path, target_sr, num_samples)


def load_ctc_vocab(path: str | Path) -> dict[int, str]:
    """Read the CTC vocabulary distributed with a checkpoint.

    Args:
        path: JSON file mapping token strings to integer IDs.

    Returns:
        The inverse mapping, indexed by token ID.
    """
    tokens = json.loads(Path(path).read_text(encoding="utf-8"))
    return {index: token for token, index in tokens.items()}


def token_ids_to_text(token_ids: list[int],
                      vocab: dict[int, str] | None = None) -> str:
    """Convert collapsed CTC token IDs to text.

    Args:
        token_ids: Token sequence after removing repeats and the CTC blank.
        vocab: ID-to-token mapping, or None for the sample vocabulary.

    Returns:
        The transcript with word separators converted to spaces.
    """
    if vocab is None:
        vocab = load_ctc_vocab(MODEL_DIR / "vocab.json")
    pieces = (vocab[token_id] for token_id in token_ids)
    return "".join(" " if token == "|" else token
                   for token in pieces
                   if not token.startswith("<")).strip()


@dataclasses.dataclass
class Wav2Vec2Config:
    """Configuration for Wav2Vec2 ASR inference.

    Attributes:
        model_path: Path to the compiled `.bin` model file.
        language: Language code for the checkpoint. Base 960h uses English.
        blank_id: CTC blank token ID (default: 0).
        vocab_path: Optional JSON vocabulary overriding the checkpoint's file.
    """

    model_path: str = str(MODEL_DIR / "wav2vec2_base_960h_1x65535_x5.bin")
    language: str = "en"
    blank_id: int = 0
    vocab_path: str | None = None


class Wav2Vec2:
    """Transcribe fixed-length audio using an X5 CTC model.

    Attributes:
        model: X5 runtime exposing the model's input and output metadata.
        num_samples: Number of waveform samples accepted by the compiled model.
        vocab: Token strings indexed by the IDs emitted by the model.
    """

    def __init__(self, config: Wav2Vec2Config):
        """Initialize the runtime wrapper and parse model metadata.

        Args:
            config: Runtime configuration including model path and inference
                parameters.
        """

        import hbm_runtime

        self.cfg = config
        if config.vocab_path:
            vocab = Path(config.vocab_path)
        else:
            vocab = Path(config.model_path).with_name("vocab.json")
            if not vocab.is_file():
                vocab = MODEL_DIR / "vocab.json"
        self.vocab = load_ctc_vocab(vocab)
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_names = self.model.output_names[self.model_name]
        self.input_shapes = self.model.input_shapes[self.model_name]
        self.waveform_shape = tuple(self.input_shapes[self.input_names[0]])
        self.num_samples = int(np.prod(self.waveform_shape))

    def set_scheduling_params(
        self,
        priority: int | None = None,
        bpu_cores: list[int] | None = None,
    ) -> None:
        """Set optional runtime scheduling parameters.

        Args:
            priority: Scheduling priority in the range `0~255`.
            bpu_cores: Optional list of BPU core indexes used for inference.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(
        self,
        audio: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Convert raw audio into the model input tensor.

        Args:
            audio: Mono 16 kHz waveform, fitted to the model input length.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.
        """

        waveform = audio_utils.wav2vec2_input(audio, self.num_samples)
        tensor = waveform.reshape(self.waveform_shape)
        return {self.model_name: {self.input_names[0]: tensor}}

    def forward(
        self, inputs: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Execute one forward pass on BPU.

        Args:
            inputs: Prepared input tensors returned by `pre_process()`.

        Returns:
            The runtime result keyed first by model name, then by tensor name.
        """

        return self.model.run(inputs)

    def post_process(
        self,
        outputs: dict,
    ) -> str:
        """Decode CTC logits into transcribed text using greedy decoding.

        Args:
            outputs: Raw runtime outputs from `forward()`.

        Returns:
            The decoded text string.

        Raises:
            ValueError: If the output has no single time-by-vocabulary matrix.
        """

        tensors = outputs.get(self.model_name, outputs)
        logits = np.atleast_2d(np.squeeze(tensors[self.output_names[0]]))
        if logits.ndim != 2:
            raise ValueError(f"expected one time-by-vocabulary CTC output, got "
                             f"{logits.shape}")

        token_ids = np.argmax(logits, axis=-1)

        # CTC collapses repeats before removing blanks.
        decoded_ids = [
            token for token, _ in itertools.groupby(token_ids)
            if token != self.cfg.blank_id
        ]

        return self._ids_to_text(decoded_ids)

    def _ids_to_text(self, token_ids: list[int]) -> str:
        """Decode IDs using this model's vocabulary.

        Args:
            token_ids: Collapsed CTC token sequence.

        Returns:
            The decoded transcript.
        """
        return token_ids_to_text(token_ids, self.vocab)

    def predict(
        self,
        audio: np.ndarray,
    ) -> str:
        """Run the complete Wav2Vec2 inference pipeline on one audio input.

        Args:
            audio: Mono 16 kHz waveform, fitted to the model input length.

        Returns:
            The transcribed text string produced by `post_process()`.
        """

        start = time.perf_counter()
        inputs = self.pre_process(audio)
        preprocessed = time.perf_counter()
        outputs = self.forward(inputs)
        inferred = time.perf_counter()
        text = self.post_process(outputs)
        decoded = time.perf_counter()

        print(f"\n[Log] Pre-process: {(preprocessed - start) * 1000:.2f} ms | "
              f"Inference: {(inferred - preprocessed) * 1000:.2f} ms | "
              f"Post-process: {(decoded - inferred) * 1000:.2f} ms")
        return text

    def __call__(
        self,
        audio: np.ndarray,
    ) -> str:
        """Provide functional-style access to `predict()`.

        Args:
            audio: Mono 16 kHz waveform, fitted to the model input length.

        Returns:
            The same transcribed text string returned by `predict()`.
        """

        return self.predict(audio)
