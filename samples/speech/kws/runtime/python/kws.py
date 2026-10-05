# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""KWS task: waveform → features → raw inference → keyword probability.

:class:`KWS` is the readable single-window model class: :meth:`preprocess`
turns one 16 kHz waveform into the owned named ``[1,373,80]`` feature tensor,
:meth:`infer` performs exactly one raw runner call, :meth:`postprocess`
validates the bound output and returns the maximum keyword probability, and
:meth:`predict` chains the three steps. The established
``pre_process``/``forward``/``post_process`` names stay thin aliases — one
implementation, two names. Feature extraction lives in
:mod:`samples.speech.kws.runtime.python.frontend`; scoring lives in
:mod:`samples.speech.kws.runtime.python.postprocess`.
"""

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

    # ------------------------------------------------------------------
    # The three pipeline stages, each public and usable on its own.
    # ------------------------------------------------------------------

    def preprocess(self, waveform, sample_rate):
        """Return owned named float32 [1,373,80] from mono float32 16 kHz samples."""
        return prepare_features(
            waveform, sample_rate, self.config, self.binding, self.frontend
        )

    def infer(self, tensors):
        """One raw runner call; no activation, reduction or dequantization."""
        return self.runner(tensors)

    def postprocess(self, raw):
        """Return maximum validated model probability; never apply another sigmoid."""
        return keyword_score(raw, self.binding)

    def predict(self, waveform, sample_rate):
        """Compose preprocess → infer → postprocess without cached context."""
        return self.postprocess(self.infer(self.preprocess(waveform, sample_rate)))

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
