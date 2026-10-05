# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One independent ASR chunk, with pure preprocessing and raw inference.

:class:`ASR` is the readable single-chunk model class: :meth:`preprocess`
turns one waveform into the owned ``[1,30000]`` float32 tensor with this
chunk's geometry, :meth:`infer` performs exactly one raw runner call,
:meth:`postprocess` decodes the bound logits, and :meth:`predict` chains the
three steps. The established ``pre_process``/``forward``/``post_process``
names stay thin aliases of the implementations above — one implementation,
two names. The audio frontend (mono mix, Fourier resampling, z-score,
padding) lives in :mod:`samples.speech.asr.runtime.python.frontend`; decoding
lives in :mod:`samples.speech.asr.runtime.python.postprocess`.
"""

from dataclasses import dataclass

from samples.speech.asr.runtime.python.frontend import (
    Config,
    PreparedChunk,
    prepare_chunk,
    validate_config,
)
from samples.speech.asr.runtime.python.decoding import validate_vocabulary
from samples.speech.asr.runtime.python.postprocess import transcribe


@dataclass(frozen=True)
class ChunkPrediction:
    """One chunk's decoded text plus this call's prepared geometry.

    Returned only by ``predict(..., return_details=True)`` so a streaming
    caller can record each chunk's own resampling geometry without re-running
    any stage. Both members are owned per call; nothing is retained on the
    model.
    """

    text: str
    prepared: PreparedChunk


class ASR:
    """One independent audio chunk per call; no state crosses chunks."""

    def __init__(
        self, runner, binding, vocabulary, config=Config(), *, decode_mode="ctc"
    ):
        validate_config(config)
        if decode_mode not in ("ctc", "legacy"):
            raise ValueError("decode-mode must be ctc or legacy")
        self.vocabulary = validate_vocabulary(vocabulary)
        if (
            len(self.vocabulary)
            != binding.metadata.output_shapes[binding.output_name][-1]
        ):
            raise ValueError("Vocabulary width differs from compiled logits")
        self.runner = runner
        self.binding = binding
        self.config = config
        self.decode_mode = decode_mode

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, waveform, sample_rate):
        """Return owned [1,30000] float32 tensor and explicit per-chunk geometry."""
        return prepare_chunk(waveform, sample_rate, self.config)

    def infer(self, tensors):
        """Perform exactly one raw runner call, without numeric transforms."""
        return self.runner(tensors)

    def postprocess(self, raw):
        """Decode one bound logit tensor; no decoder state crosses chunks."""
        return transcribe(raw, self.binding, self.vocabulary, self.decode_mode)

    def predict(self, waveform, sample_rate, *, return_details=False):
        """Compose preprocessing, raw inference and decoding for one audio chunk.

        The default return is the decoded text. With ``return_details=True``
        the call returns a :class:`ChunkPrediction` holding the same text and
        this call's prepared chunk (owned tensor, valid sample count and
        source geometry), so per-chunk evidence comes from the single
        execution this method performs. No state is retained on the model.
        """
        prepared = self.preprocess(waveform, sample_rate)
        text = self.postprocess(
            self.infer({self.binding.input_name: prepared.tensor})
        )
        if return_details:
            return ChunkPrediction(text, prepared)
        return text

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin aliases
    # of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, waveform, sample_rate):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(waveform, sample_rate)

    def forward(self, tensors):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(tensors)

    def post_process(self, raw):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(raw)
