# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One independent ASR chunk, with pure preprocessing and raw inference."""

from samples.speech.asr.runtime.python.frontend import (
    Config,
    prepare_chunk,
    validate_config,
)
from samples.speech.asr.runtime.python.decoding import validate_vocabulary
from samples.speech.asr.runtime.python.postprocess import transcribe


class ASR:
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

    def pre_process(self, waveform, sample_rate):
        """Return owned [1,30000] float32 tensor and explicit per-chunk geometry."""
        return prepare_chunk(waveform, sample_rate, self.config)

    def forward(self, tensors):
        """Perform exactly one raw runner call, without numeric transforms."""
        return self.runner(tensors)

    def post_process(self, raw):
        """Decode one bound logit tensor; no decoder state crosses chunks."""
        return transcribe(raw, self.binding, self.vocabulary, self.decode_mode)

    def predict(self, waveform, sample_rate):
        """Compose preprocessing, raw inference and decoding for one audio chunk."""
        prepared = self.pre_process(waveform, sample_rate)
        return self.post_process(
            self.forward({self.binding.input_name: prepared.tensor})
        )
