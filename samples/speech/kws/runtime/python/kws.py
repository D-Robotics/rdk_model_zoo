# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""KWS task: waveform → features → raw inference → keyword probability."""

from samples.speech.kws.runtime.python.frontend import (
    Config,
    validate_config,
    prepare_features,
)
from samples.speech.kws.runtime.python.postprocess import keyword_score


class KWS:
    """Owned raw results through the shared runner; no file I/O or SDK imports."""

    def __init__(self, runner, binding, config=Config(), *, frontend=None):
        validate_config(config)
        self.runner = runner
        self.binding = binding
        self.config = config
        self.frontend = frontend

    def pre_process(self, waveform, sample_rate):
        """Return owned named float32 [1,373,80] from mono float32 16 kHz samples."""
        return prepare_features(
            waveform, sample_rate, self.config, self.binding, self.frontend
        )

    def forward(self, tensors):
        """One raw runner call; no activation, reduction or dequantization."""
        return self.runner(tensors)

    def post_process(self, raw):
        """Return maximum validated model probability; never apply another sigmoid."""
        return keyword_score(raw, self.binding)

    def predict(self, waveform, sample_rate):
        """Compose pre_process, forward and post_process without cached context."""
        return self.post_process(self.forward(self.pre_process(waveform, sample_rate)))
