# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""KWS task: waveform → features → raw inference → keyword probability.

:class:`KWS` is the readable single-window model class: :meth:`preprocess`
turns one 16 kHz waveform into the owned named ``[1,373,80]`` feature tensor,
:meth:`infer` performs exactly one raw runner call, :meth:`postprocess`
validates the bound output and returns the maximum keyword probability, and
:meth:`predict` chains the three steps. The established
``pre_process``/``forward``/``post_process`` names stay thin aliases — one
implementation, two names. This file also holds the fixed MDTC feature
frontend (PaddleAudio fbank imported lazily at execution), the published
tensor binding and the raw runner construction (:class:`RuntimeModelRunner`),
so the model file is the complete task boundary and callers never assemble
runners themselves. Published asset selection/listing and audio-file
delivery live in ``cli.py``.
"""

from dataclasses import dataclass

import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import apply_output_transform, validate_scale_quantization
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import SingleArrayRunner
from samples.speech.kws.runtime.python.cli import Selection, resolve_selection

# ======================================================================
# Fixed MDTC feature protocol, independent of file I/O and board SDK.
# ======================================================================

@dataclass(frozen=True)
class Config:
    audio_maxlen: int = 60000
    frame_shift: int = 10
    frame_length: int = 25
    n_mels: int = 80


def validate_config(config):
    values = (
        config.audio_maxlen,
        config.frame_shift,
        config.frame_length,
        config.n_mels,
    )
    if any(type(v) is not int for v in values) or values != (60000, 10, 25, 80):
        raise ValueError(
            "Published KWS requires audio-maxlen=60000, frame-shift=10, frame-length=25, n-mels=80"
        )


def prepare_waveform(waveform, sample_rate, config):
    validate_config(config)
    if type(sample_rate) is not int or sample_rate != 16000:
        raise ValueError("KWS requires 16000 Hz audio; resampling is not implicit")
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.ndim != 1
        or waveform.size == 0
        or waveform.dtype != np.float32
        or not np.isfinite(waveform).all()
    ):
        raise ValueError("Expected nonempty finite mono float32 waveform [N]")
    if np.any(np.abs(waveform) > 1):
        raise ValueError("Expected normalized PCM amplitudes in [-1,1]")
    output = np.zeros((1, config.audio_maxlen), np.float32)
    count = min(waveform.size, config.audio_maxlen)
    output[0, :count] = waveform[:count]
    return output


def paddle_fbank(waveform, config):
    """Use the source PaddleAudio frontend lazily, with explicit source defaults."""
    import paddle
    from paddleaudio.compliance.kaldi import fbank

    return fbank(
        waveform=paddle.to_tensor(waveform),
        sr=16000,
        frame_shift=config.frame_shift,
        frame_length=config.frame_length,
        n_mels=config.n_mels,
    ).numpy()


def prepare_features(waveform, sample_rate, config, binding, frontend=None):
    samples = prepare_waveform(waveform, sample_rate, config)
    features = (frontend or paddle_fbank)(samples, config)
    if (
        not isinstance(features, np.ndarray)
        or features.shape != (373, 80)
        or features.dtype != np.float32
        or not np.isfinite(features).all()
    ):
        raise ValueError("Frontend must produce finite float32 [373,80] features")
    tensor = np.array(features[None], dtype=np.float32, order="C", copy=True)
    if tensor.shape != binding.metadata.input_shapes[binding.input_name]:
        raise ValueError("Extracted feature shape differs from model metadata")
    return {binding.input_name: tensor}

# ======================================================================
# Published SDK tensor checks; no SDK import.
# ======================================================================

@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata
    input_name: str
    output_name: str

    @property
    def model_name(self):
        return self.metadata.model_name


def bind_model(selection, metadata):
    expected = resolve_selection(
        selection.target,
        asset_id=selection.asset.reference,
        model_path=selection.model_path if selection.explicit_model_path else None,
    )
    if selection != expected:
        raise ValueError("Selection does not match the published identity/path")
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if (
        meta.model_names != (meta.model_name,)
        or len(meta.input_names) != 1
        or len(meta.output_names) != 1
    ):
        raise MetadataMismatchError("KWS requires exactly one model/input/output")
    name, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(name) != (1, 373, 80)
        or meta.input_dtypes.get(name) != "float32"
    ):
        raise MetadataMismatchError(
            "KWS input must be float32 [1,373,80] for the fixed frontend"
        )
    shape = meta.output_shapes.get(out, ())
    if not shape or shape[0] != 1 or any(type(n) is not int or n <= 0 for n in shape):
        raise MetadataMismatchError(
            "KWS score output requires finite positive dimensions and batch one"
        )
    dtype = meta.output_dtypes.get(out)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError("Unsupported KWS output dtype")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(out), shape)
    return Binding(selection, meta, name, out)

# ======================================================================
# The readable task boundary.
# ======================================================================


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
