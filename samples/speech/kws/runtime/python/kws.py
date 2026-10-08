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
:mod:`samples.speech.kws.runtime.python.frontend`; the raw runner
construction (:class:`RuntimeModelRunner`), the model-owned loader
:meth:`KWS.from_model` and the probability scoring are consolidated here so
the model file holds the complete task boundary and callers never assemble
runners themselves.
"""

import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import apply_output_transform
from utils.py_utils.single_array_runner import SingleArrayRunner
from samples.speech.kws.runtime.python.frontend import (
    Config,
    validate_config,
    prepare_features,
)
from samples.speech.kws.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    """KWS binding over the shared raw single-array transport."""

    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: ((1, 373, 80), "float32"),
            task_name="KWS",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )


class KWS:
    """Owned raw results through the shared runner; no file I/O or SDK imports."""

    def __init__(self, runner, binding, config=Config(), *, frontend=None):
        validate_config(config)
        self.runner = runner
        self.binding = binding
        self.config = config
        self.frontend = frontend

    @classmethod
    def from_model(cls, selection, config=Config(), *, frontend=None,
                   runtime=None, runtime_factory=None):
        """Construct the loaded KWS task for one resolved selection.

        The model class owns the low-level assembly: it creates the raw
        runner, loads and binds the model (running the board/asset gates
        unless a ``runtime`` or ``runtime_factory`` host seam is supplied)
        and returns a task whose ``predict`` is ready to call.

        Args:
            selection: Selection from ``resolve_selection``; its target must
                match the executing board on real execution.
            config: Fixed frontend Config; see __init__.
            frontend: Optional feature-extraction callable overriding the
                PaddleAudio frontend (host-test seam).
            runtime: Optional injected SDK runtime (host-test seam).
            runtime_factory: Optional SDK factory callable (host-test seam).

        Returns:
            KWS: Loaded task constructed with the pure injected constructor.

        Raises:
            ValueError: Board identity, publication or frontend settings
                violate the contract.
            MetadataMismatchError: SDK metadata differs from the KWS contract.
            RuntimeError: The board SDK is unavailable or loading fails.
        """
        runner = RuntimeModelRunner(
            selection, runtime=runtime, runtime_factory=runtime_factory
        )
        return cls(runner, runner.load(), config, frontend=frontend)

    @property
    def metadata(self):
        """Bound SDK metadata of the loaded model (evidence for reports)."""
        return self.binding.metadata

    def set_scheduling_params(self, *, priority=None, bpu_cores=None) -> None:
        """Apply scheduling options to the loaded board runtime.

        Args:
            priority: Optional integer in [0, 255]; None leaves it unchanged.
            bpu_cores: Optional list of nonnegative BPU core indexes.

        Returns:
            None.

        Raises:
            ValueError: Priority or a core index is out of range.
            RuntimeError: The SDK cannot apply the scheduling options.
        """
        self.runner.set_scheduling_params(priority=priority, bpu_cores=bpu_cores)

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
        meta = self.binding.metadata
        name = self.binding.output_name
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != meta.output_shapes[name]
            or raw.dtype != np.dtype(meta.output_dtypes[name])
            or not np.isfinite(raw).all()
        ):
            raise ValueError("KWS raw output differs from bound shape/dtype/finite values")
        mode = "raw_f32" if raw.dtype == np.float32 else "dequant"
        values = apply_output_transform(mode, {name: raw}, meta.output_quants)[name]
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(
                "KWS expects already activated probabilities in [0,1], not logits"
            )
        return float(np.max(values))

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
