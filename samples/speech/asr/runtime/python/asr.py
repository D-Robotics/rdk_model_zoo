# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""One independent ASR chunk, with pure preprocessing and raw inference.

:class:`ASR` is the readable single-chunk model class: :meth:`preprocess`
turns one waveform into the owned ``[1,30000]`` float32 tensor with this
chunk's geometry, :meth:`infer` performs exactly one raw runner call,
:meth:`postprocess` decodes the bound logits, and :meth:`predict` chains the
three steps. The established ``pre_process``/``forward``/``post_process``
names stay thin aliases of the implementations above — one implementation,
two names. This file also holds the audio frontend (mono mix, Fourier
resampling, z-score, padding), the greedy CTC decoders (legacy mode keeps
repeats and ``|`` verbatim), the published tensor binding and the raw runner
construction (:class:`RuntimeModelRunner`), so the model file is the
complete task boundary and callers never assemble runners themselves.
Published asset selection/listing, audio-file streaming and vocabulary
delivery live in ``cli.py``.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from utils.py_utils.platforms import require_execution_target
from utils.py_utils.quantization import (
    apply_output_transform,
    dequantize_tensor,
    validate_scale_quantization,
)
from utils.py_utils.runtime_meta import RuntimeMetadata, MetadataMismatchError
from utils.py_utils.single_array_runner import SingleArrayRunner
from samples.speech.asr.runtime.python.cli import Selection, resolve_selection

# ======================================================================
# Audio frontend: pure per-chunk mono mix, Fourier resampling, z-score
# and padding operations.
# ======================================================================

@dataclass(frozen=True)
class Config:
    audio_maxlen: int = 30000
    new_rate: int = 16000


@dataclass(frozen=True)
class PreparedChunk:
    tensor: np.ndarray
    valid_samples: int
    source_samples: int
    source_rate: int


def validate_config(config):
    if (
        type(config.audio_maxlen) is not int
        or type(config.new_rate) is not int
        or (config.audio_maxlen, config.new_rate) != (30000, 16000)
    ):
        raise ValueError("Published ASR uses 30000 target samples at 16000 Hz")


def source_chunk_size(sample_rate, config=Config()):
    validate_config(config)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    return (config.audio_maxlen * sample_rate + config.new_rate - 1) // config.new_rate


def prepare_chunk(waveform, sample_rate, config=Config()):
    limit = source_chunk_size(sample_rate, config)
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.ndim not in (1, 2)
        or waveform.dtype != np.float32
        or not 0 < len(waveform) <= limit
        or (waveform.ndim == 2 and waveform.shape[1] < 1)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError(
            "Expected finite float32 [frames] or [frames,channels] within one source chunk"
        )
    mono = waveform.mean(axis=1) if waveform.ndim == 2 else waveform.copy()
    if sample_rate != config.new_rate:
        from scipy.signal import resample

        count = round(len(mono) * config.new_rate / sample_rate)
        if count < 1:
            raise ValueError("Chunk duration is too short to form one target sample")
        mono = resample(mono, count)
    normalized = (mono - np.mean(mono)) / np.sqrt(np.var(mono) + 1e-5)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalization produced nonfinite values")
    count = min(len(normalized), config.audio_maxlen)
    tensor = np.zeros((1, config.audio_maxlen), np.float32)
    tensor[0, :count] = normalized[:count]
    return PreparedChunk(tensor, count, len(waveform), sample_rate)

# ======================================================================
# Greedy CTC decoders: float32 or exact comparison precision, plus the
# archived concatenate-only legacy mode.
# ======================================================================

def validate_vocabulary(vocabulary):
    if (
        not isinstance(vocabulary, (list, tuple))
        or not vocabulary
        or any(not isinstance(token, str) or not token for token in vocabulary)
        or vocabulary[0] != "<pad>"
        or len(set(vocabulary)) != len(vocabulary)
    ):
        raise ValueError("Expected unique nonempty ordered tokens with <pad> at ID 0")
    return tuple(vocabulary)


def decode_ids(ids, vocabulary, mode="legacy"):
    tokens = validate_vocabulary(vocabulary)
    if mode not in ("ctc", "legacy"):
        raise ValueError("decode-mode must be ctc or legacy")
    output = []
    previous = None
    for token in ids:
        if (
            isinstance(token, (bool, np.bool_))
            or not isinstance(token, Integral)
            or not 0 <= token < len(tokens)
        ):
            raise ValueError("Token IDs must be integers within the vocabulary")
        if token != 0 and (mode == "legacy" or token != previous):
            output.append(tokens[token])
        previous = token
    text = "".join(output)
    # Wav2Vec2 CTC vocabulary uses | as the word delimiter, not a glyph.
    # Preserve the archived concatenate-only representation in legacy mode.
    return text.replace("|", " ").strip() if mode == "ctc" else text


def decode_logits(logits, vocabulary, mode="legacy"):
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < 1
        or logits.shape[2] != len(tokens)
        or logits.dtype != np.float32
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float32 logits [1,T,vocabulary_size], T > 0")
    return decode_ids(np.argmax(logits[0], axis=-1), tokens, mode)


def decode_exact_logits(logits, vocabulary, mode="legacy"):
    """Greedy decode of float64 logits carrying exact integer affine scores.

    Integer SCALE outputs dequantized with ``dequantize_tensor(dtype="float64")``
    keep distinct raw scores distinct through argmax; the float32
    :func:`decode_logits` rounds adjacent int32 magnitudes such as 2**24 and
    2**24 + 1 into artificial ties.  Genuinely equal scores still tie to the
    lowest ID exactly like :func:`decode_logits`.
    """
    tokens = validate_vocabulary(vocabulary)
    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] < 1
        or logits.shape[2] != len(tokens)
        or logits.dtype != np.float64
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Expected finite float64 logits [1,T,vocabulary_size], T > 0")
    return decode_ids(np.argmax(logits[0], axis=-1), tokens, mode)

# ======================================================================
# Published SDK tensor checks; no SDK import.
# ======================================================================

VOCABULARY_SIZE = 3503


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
        raise MetadataMismatchError("ASR requires exactly one model/input/output")
    name, out = meta.input_names[0], meta.output_names[0]
    if (
        meta.input_shapes.get(name) != (1, 30000)
        or meta.input_dtypes.get(name) != "float32"
    ):
        raise MetadataMismatchError(
            "ASR input must be float32 [1,30000] for the fixed frontend"
        )
    shape = meta.output_shapes.get(out, ())
    if (
        len(shape) != 3
        or shape[0] != 1
        or type(shape[1]) is not int
        or shape[1] <= 0
        or shape[2] != VOCABULARY_SIZE
    ):
        raise MetadataMismatchError("ASR logits must be [1,T,3503], T > 0")
    dtype = meta.output_dtypes.get(out)
    if dtype not in ("float32", "int8", "uint8", "int16", "int32"):
        raise MetadataMismatchError("Unsupported ASR output dtype")
    if dtype != "float32":
        validate_scale_quantization(meta.output_quants.get(out), shape)
    return Binding(selection, meta, name, out)

# ======================================================================
# The readable task boundary.
# ======================================================================


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
