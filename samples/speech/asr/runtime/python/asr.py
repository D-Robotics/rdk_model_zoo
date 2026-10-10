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
padding) lives in :mod:`samples.speech.asr.runtime.python.frontend`; the
greedy CTC decoders live in
:mod:`samples.speech.asr.runtime.python.decoding`. The raw runner
construction (:class:`RuntimeModelRunner`), the model-owned loader
:meth:`ASR.from_model` and the physical output validation/dispatch
(:func:`transcribe`) are consolidated here so the model file holds the
complete task boundary and callers never assemble runners themselves.
"""

from dataclasses import dataclass

import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import apply_output_transform, dequantize_tensor
from utils.py_utils.single_array_runner import SingleArrayRunner
from samples.speech.asr.runtime.python.frontend import (
    Config,
    PreparedChunk,
    prepare_chunk,
    validate_config,
)
from samples.speech.asr.runtime.python.decoding import (
    decode_exact_logits,
    decode_logits,
    validate_vocabulary,
)
from samples.speech.asr.runtime.python.model_binding import bind_model


class RuntimeModelRunner(SingleArrayRunner):
    """ASR binding over the shared raw single-array transport."""

    def __init__(self, selection, *, runtime_factory=None, runtime=None):
        super().__init__(
            selection,
            binding_loader=bind_model,
            physical_input=lambda binding: ((1, 30000), "float32"),
            task_name="ASR",
            runtime_factory=runtime_factory,
            runtime=runtime,
            execution_target_gate=require_execution_target,
        )


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


def transcribe(raw, binding, vocabulary, mode):
    """Validate one raw logit tensor and decode it with the selected mode."""
    name = binding.output_name
    meta = binding.metadata
    if (
        not isinstance(raw, np.ndarray)
        or raw.shape != meta.output_shapes[name]
        or raw.dtype != np.dtype(meta.output_dtypes[name])
        or not np.isfinite(raw).all()
    ):
        raise ValueError("ASR raw logits differ from bound shape/dtype/values")
    if raw.dtype == np.float32:
        logits = apply_output_transform("raw_f32", {name: raw}, meta.output_quants)[name]
        return decode_logits(np.asarray(logits, dtype=np.float32), vocabulary, mode)
    quant = meta.output_quants.get(name)
    if quant is None:
        raise ValueError(
            f"Integer ASR output {name!r} has no quantization descriptor in "
            "output_quants; F32 artifacts use the raw_f32 transform."
        )
    # Integer affine scores must keep their ordering through argmax: float32
    # rounds adjacent int32 magnitudes (e.g. 2**24 vs 2**24 + 1) into artificial
    # ties, so dequantization runs at float64 and the exact decoder takes over.
    return decode_exact_logits(
        dequantize_tensor(raw, quant, dtype="float64"), vocabulary, mode
    )


class ASR:
    """One independent audio chunk per call; no state crosses chunks."""

    def __init__(
        self, runner, binding, vocabulary, config=Config(), *, decode_mode="legacy"
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

    @classmethod
    def from_model(
        cls, selection, vocabulary, config=Config(), *, decode_mode="legacy",
        runtime=None, runtime_factory=None,
    ):
        """Construct the loaded ASR task for one resolved selection.

        The model class owns the low-level assembly: it creates the raw
        runner, loads and binds the model (running the board/asset gates
        unless a ``runtime`` or ``runtime_factory`` host seam is supplied)
        and returns a task whose ``predict`` is ready to call.

        Args:
            selection: Selection from ``resolve_selection``; its target must
                match the executing board on real execution.
            vocabulary: Ordered token sequence; see __init__.
            config: Fixed frontend Config; see __init__.
            decode_mode: ``ctc`` or ``legacy`` decoding.
            runtime: Optional injected SDK runtime (host-test seam).
            runtime_factory: Optional SDK factory callable (host-test seam).

        Returns:
            ASR: Loaded task constructed with the pure injected constructor.

        Raises:
            ValueError: Board identity, publication, vocabulary or decode
                settings violate the contract.
            MetadataMismatchError: SDK metadata differs from the ASR contract.
            RuntimeError: The board SDK is unavailable or loading fails.
        """
        runner = RuntimeModelRunner(
            selection, runtime=runtime, runtime_factory=runtime_factory
        )
        return cls(runner, runner.load(), vocabulary, config, decode_mode=decode_mode)

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
