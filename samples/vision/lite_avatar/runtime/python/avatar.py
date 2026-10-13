# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Four-model LiteAvatar composition: stages, binding, runners and rendering.

``avatar.py`` owns the model side end to end for the audio-driven talking-
avatar pipeline: the per-model stages (``AudioEncoderStage`` /
``Audio2MouthStage`` / ``FaceGenEncoderStage`` / ``FaceGenDecoderStage`` with
the canonical ``preprocess``/``infer``/``postprocess`` spellings), the
published tensor contracts and stage-IO binding, the four lazy raw runners,
the audio2mouth window/crossfade algorithm, the LZ4 background cache and the
compositing renderer, and the application-level :class:`LiteAvatarPipeline`.

The pipeline is adapted from HumanAIGC/lite-avatar (MIT) and its validated
RDK S100 adaptation: waveform -> LFR/CMVN fbank features -> Paraformer-derived
audio encoder hidden states -> time interpolation -> audio2mouth parameters
-> face-generator decoder composites over cached background frames.  The
decoder artifact mixes INT16 BPU convolutions with an FP32 CPU InstanceNorm
subgraph; that is a property of the compiled HBM, not of this code.

Five concepts (per the sample standards):

- **Input**: :class:`AvatarInput` — finite waveform samples (mono or
  multichannel), the source sample rate, nothing else.  Frame count is
  derived, never guessed.
- **Tensors**: per-stage ``dict[str, np.ndarray]`` feeds documented in
  :data:`INPUTS` (exact shapes/dtypes; all four golden S100 artifacts take
  float32 featuremap inputs and expose float32 outputs).
- **Context**: :class:`AvatarContext` — per-call geometry: valid feature
  frames ``feature_frames``, output ``frame_count`` at 30 fps.  Stage
  ``preprocess`` returns it inside :class:`PreparedInput`.
- **RawOutputs**: the owned ``dict[str, np.ndarray]`` mapping returned by the
  shared :class:`~utils.py_utils.single_array_runner.NamedArrayRunner` after
  structural/metadata validation (exact name sets, shapes, dtypes).
- **Result**: :class:`AvatarResult` — rendered-clip metadata plus a bounded
  owned-frame iterator.  Video encoding and file delivery belong to the CLI.

Session semantics (stateful generative task): ``predict`` runs the two audio
models, then *releases* them (their ~257 MiB of artifacts and workspaces are
dead weight while rendering, and the S100 has 2.8 GiB RAM) before the
renderer lazily loads the two face-generator models and caches reference
pyramids.  Calling ``predict`` again reloads the audio models from disk;
reference caches are kept.  Audio file reading, published selection and
video writing live in ``cli.py``/``main.py``; the deterministic feature
arithmetic lives in ``frontend.py``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import gc
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from utils.py_utils.quantization import OutputTransformError, dequantize_tensor
from utils.py_utils.runtime_meta import MetadataMismatchError, RuntimeMetadata
from utils.py_utils.single_array_runner import NamedArrayRunner
from samples.vision.lite_avatar.runtime.python.cli import STAGES, Selection
from samples.vision.lite_avatar.runtime.python.frontend import (
    AvatarFrontend,
    prepare_waveform,
)

# ======================================================================
# Published tensor contracts (names/shapes verified against the four
# golden S100 artifacts; see docs/release/s/models.yaml).
# ======================================================================

#: Output frame rate of the rendered video (the a2m/facegen cadence).
FRAME_RATE = 30
#: Audio2mouth sliding window (1 s) and crossfade (0.2 s), in frames.
A2M_WINDOW = 30
A2M_CROSSFADE = 6
#: Maximum audio length: the encoder artifact is fixed at 400 LFR frames
#: (~24 s of audio).  Longer input is rejected, never silently truncated.
MAX_FEATURE_FRAMES = 400

INPUTS = {
    "audio_encoder": {"feats": (("feats",), (1, 400, 560), "float32")},
    "audio2mouth": {"input_au": (("input_au",), (1, 30, 50, 512), "float32")},
    "facegen_encoder": {"image": (("image",), (1, 3, 384, 384), "float32")},
    "facegen_decoder": {
        "f0": (("f0",), (1, 3, 384, 384), "float32"),
        "f1": (("f1",), (1, 16, 192, 192), "float32"),
        "f2": (("f2",), (1, 32, 96, 96), "float32"),
        "f3": (("f3",), (1, 64, 48, 48), "float32"),
        "mouth_param": (("mouth_param",), (1, 32), "float32"),
    },
}

OUTPUTS = {
    "audio_encoder": {
        "hidden_states": (("hidden_states",), (1, 400, 50, 512), "float32")
    },
    # ``feat`` is the folded model's auxiliary feature output; it is part of
    # the validated output set and deliberately unused by the renderer.
    "audio2mouth": {
        "output": (("output",), (1, 30, 32), "float32"),
        "feat": (("feat",), (1, 30, 256), "float32"),
    },
    "facegen_encoder": {
        "f0": (("f0",), (1, 3, 384, 384), "float32"),
        "f1": (("f1",), (1, 16, 192, 192), "float32"),
        "f2": (("f2",), (1, 32, 96, 96), "float32"),
        "f3": (("f3",), (1, 64, 48, 48), "float32"),
    },
    "facegen_decoder": {
        "mouth_image": (("mouth_image",), (1, 3, 384, 384), "float32")
    },
}


@dataclass(frozen=True)
class Binding:
    selection: Selection
    metadata: RuntimeMetadata
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]

    @property
    def model_name(self):
        return self.metadata.model_name


def _roles(names, shapes, dtypes, contracts):
    if len(set(names)) != len(names):
        raise MetadataMismatchError("Physical tensor names must be unique")
    roles = {}
    for role, (aliases, shape, dtype) in contracts.items():
        matches = [name for name in names if name in aliases]
        if len(matches) != 1:
            raise MetadataMismatchError(f"Expected one exact physical name for {role}")
        name = matches[0]
        if shapes.get(name) != shape or dtypes.get(name) != dtype:
            raise MetadataMismatchError(f"{name} must be {dtype} {shape}")
        roles[role] = name
    if set(names) != set(roles.values()):
        raise MetadataMismatchError("Unexpected extra physical tensors")
    return roles


def validate_selection(selection):
    """Reject stage/asset/path mismatches before SDK construction.

    Identity (filename, reference, format, url) must match the published row
    exactly.  The SHA-256 digest is either the published one or — only when
    the selection is marked ``digest_source="local"`` — a digest recorded in
    a user-provided rebuild manifest; any other asset difference is rejected.
    """

    from dataclasses import replace as _replace

    from samples.vision.lite_avatar.runtime.python.cli import resolve_selections

    if selection.stage not in STAGES:
        raise ValueError("Unknown LiteAvatar stage")
    expected = resolve_selections(selection.target)[STAGES.index(selection.stage)]
    if selection.digest_source == "local":
        identity = _replace(selection.asset, sha256=expected.asset.sha256)
        if identity != expected.asset:
            raise ValueError(
                "Local rebuild selection must keep the published asset "
                "identity and change only the recorded digest"
            )
    elif selection.asset != expected.asset:
        raise ValueError("Selection does not match the declared stage publication")
    if not selection.explicit_model_path and selection.model_path != expected.model_path:
        raise ValueError("Selection does not match the declared stage path")


def bind_model(selection, metadata):
    """Validate publication identity, one model and every exposed I/O tensor."""

    validate_selection(selection)
    meta = (
        metadata
        if isinstance(metadata, RuntimeMetadata)
        else RuntimeMetadata.from_mapping(metadata)
    )
    if meta.model_names != (meta.model_name,):
        raise MetadataMismatchError(
            "Each LiteAvatar artifact must expose exactly one model"
        )
    inputs = _roles(
        meta.input_names, meta.input_shapes, meta.input_dtypes, INPUTS[selection.stage]
    )
    outputs = _roles(
        meta.output_names, meta.output_shapes, meta.output_dtypes, OUTPUTS[selection.stage]
    )
    return Binding(selection, meta, inputs, outputs)


def physical_inputs(binding):
    return {
        name: (binding.metadata.input_shapes[name], binding.metadata.input_dtypes[name])
        for name in binding.metadata.input_names
    }


# ======================================================================
# Shared tensor helpers.
# ======================================================================


def tensor(value, shape, name, dtype="float32"):
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype(dtype)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite {dtype} {shape}")
    return np.array(value, order="C", copy=True)


@dataclass(frozen=True)
class OwnedQuantDescriptor:
    """Owned copy of one output's quantization facts.

    SDK quant descriptors (``QuantParams``) belong to the loaded runtime:
    once a runner unloads, they are borrowed memory that must not be held
    by long-lived stages.  This record keeps exactly the four facts the
    shared dequantization chain reads — with attribute access, so
    :func:`utils.py_utils.quantization.dequantize_tensor` consumes it
    unchanged — as plain owned data: the enum-like type tag as ``str``,
    ``scale``/``zero_point`` as copied NumPy arrays, ``axis`` as int or
    None.  ``None`` scale/zero-point marks a descriptor whose type never
    reaches the arithmetic (non-SCALE types pass through unchanged).
    """

    quant_type: str
    scale: np.ndarray | None
    zero_point: np.ndarray | None
    axis: int | None


def detach_quants(quants):
    """Detach one per-output quant mapping into owned descriptors.

    Mirrors the shared evidence projection's descriptor handling
    (:func:`utils.py_utils.runtime_meta._evidence_value`): type tag reduced
    to its enum-like name, scalars/arrays copied, never a deepcopy of the
    SDK object.  ``None`` descriptors stay ``None`` (the F32/NONE golden
    boundary), already-owned records pass through, and a foreign shape
    (plain mapping, or no ``quant_type`` surface) fails loudly instead of
    silently dropping dequantization later.
    """

    if quants is None:
        return None
    owned: dict[str, OwnedQuantDescriptor | None] = {}
    for name, descriptor in dict(quants).items():
        if descriptor is None or isinstance(descriptor, OwnedQuantDescriptor):
            owned[str(name)] = descriptor
            continue
        quant_type = getattr(descriptor, "quant_type", None)
        if quant_type is None:
            raise MetadataMismatchError(
                f"Quant descriptor for output {name!r} exposes no quant_type; "
                "pass the SDK descriptor or an OwnedQuantDescriptor"
            )
        scale = getattr(descriptor, "scale", None)
        zero_point = getattr(descriptor, "zero_point", None)
        axis = getattr(descriptor, "axis", None)
        type_name = str(getattr(quant_type, "name", quant_type))
        if type_name == "SCALE" and (scale is None or zero_point is None):
            raise MetadataMismatchError(
                f"SCALE descriptor for output {name!r} lacks scale/zero_point"
            )
        owned[str(name)] = OwnedQuantDescriptor(
            quant_type=type_name,
            scale=None if scale is None else np.array(scale, copy=True),
            zero_point=None
            if zero_point is None
            else np.array(zero_point, copy=True),
            axis=None if axis is None else int(axis),
        )
    return owned


def float_outputs(raw, quants):
    """Return owned float32 stage outputs from validated raw runner values.

    The four golden S100 artifacts already expose F32 outputs (quant_type
    NONE at the ``hbm_runtime`` boundary).  Integer outputs from other
    builds are dequantized through the shared SCALE chain and *must* carry a
    descriptor; there is no first-output or silent-passthrough fallback.
    """

    values = {}
    for name, value in raw.items():
        array = np.asarray(value)
        if array.dtype == np.dtype("float32"):
            values[name] = array
            continue
        descriptor = quants.get(str(name)) if quants is not None else None
        if descriptor is None:
            raise OutputTransformError(
                f"Quantized output {name!r} has no descriptor; F32 artifacts "
                "need no dequantization."
            )
        values[name] = np.asarray(dequantize_tensor(array, descriptor), dtype=np.float32)
        if not np.isfinite(values[name]).all():
            raise OutputTransformError(f"Dequantized output {name!r} is non-finite")
    return values


# ======================================================================
# Business input/result records (the Input and Result concepts).
# ======================================================================


@dataclass(frozen=True)
class AvatarInput:
    """One clip: finite waveform samples and their source sample rate.

    ``waveform`` is ``[samples]`` or ``[samples, channels]`` float32/float64
    (soundfile-style); multichannel input keeps its first channel only.
    """

    waveform: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class AvatarContext:
    """Per-call geometry: valid feature frames and output frame count."""

    feature_frames: int
    frame_count: int


@dataclass(frozen=True)
class PreparedInput:
    tensors: dict[str, np.ndarray]
    context: AvatarContext | None = None


class StageError(ValueError):
    """Stage/operation attribution with the original exception chained."""

    def __init__(self, stage, operation, error):
        self.stage, self.operation = stage, operation
        super().__init__(f"{stage} {operation} failed: {error}")


def attributed(label):
    """Decorate one stage operation; pins the established error wording."""

    def decorate(operation):
        @functools.wraps(operation)
        def execute(self, *args, **kwargs):
            try:
                return operation(self, *args, **kwargs)
            except Exception as error:
                raise StageError(self.stage, label, error) from error

        return execute

    return decorate


# ======================================================================
# Public per-model stages.
# ======================================================================


class RawStage:
    """One raw model call with exact named I/O and owned float outputs."""

    stage = "stage"

    def __init__(self, runner, inputs, outputs, quants):
        if not callable(runner):
            raise TypeError("runner must be callable")
        self.runner = runner
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        # Stages outlive loaded SDK runtimes (audio release, renderer-init
        # handoff): the descriptors are detached into owned records here so
        # no borrowed SDK object is retained (idempotent for owned input).
        self.quants = detach_quants(quants)

    def _validate(self, values, contracts):
        if not isinstance(values, Mapping):
            raise ValueError("Expected a named tensor mapping")
        if set(values) != set(contracts):
            raise ValueError(f"Expected exactly tensor names {tuple(contracts)}")
        return {
            name: tensor(values[name], shape, name, dtype)
            for name, (shape, dtype) in contracts.items()
        }

    @attributed(label="infer")
    def infer(self, tensors):
        """Validate physical inputs, call only this model, return float outputs.

        Raw integer outputs (non-golden builds) are dequantized through the
        shared SCALE chain; activation/compositing stays in postprocess.
        """

        feed = self._validate(tensors, self.inputs)
        raw = self.runner(feed)
        if set(raw) != set(self.outputs):
            raise MetadataMismatchError(
                f"{self.stage} runner returned unexpected output names {tuple(raw)}"
            )
        for name, (shape, _dtype) in self.outputs.items():
            if np.asarray(raw[name]).shape != shape:
                raise MetadataMismatchError(
                    f"{self.stage} output {name!r} must be {shape}"
                )
        return float_outputs(raw, self.quants)


class AudioEncoderStage(RawStage):
    """Features ``[1,400,560]`` -> encoder hidden states ``[T,50,512]``."""

    stage = "audio_encoder"

    def __init__(self, runner, input_name, output_name, quants):
        self.input_name, self.output_name = input_name, output_name
        super().__init__(
            runner,
            {input_name: ((1, 400, 560), "float32")},
            {output_name: ((1, 400, 50, 512), "float32")},
            quants,
        )

    @attributed(label="preprocess")
    def preprocess(self, features):
        """Own padded finite float32 [1,400,560] features; no audio I/O.

        Features longer than 400 LFR frames (~24 s) are rejected: the
        published encoder artifact has a fixed 400-frame window and the
        validated pipeline never extends it silently.
        """

        if (
            not isinstance(features, np.ndarray)
            or features.ndim != 2
            or features.shape[1] != 560
            or features.dtype != np.dtype("float32")
            or not np.isfinite(features).all()
        ):
            raise ValueError("features must be finite float32 [T,560]")
        frames = features.shape[0]
        if not 1 <= frames <= MAX_FEATURE_FRAMES:
            raise ValueError(
                f"Audio maps to {frames} LFR frames; the encoder window is "
                f"1..{MAX_FEATURE_FRAMES} frames (~24 s at 16 kHz)"
            )
        window = np.zeros((1, MAX_FEATURE_FRAMES, 560), np.float32)
        window[0, :frames] = features
        return PreparedInput({self.input_name: window}, AvatarContext(frames, 0))

    @attributed(label="postprocess")
    def postprocess(self, outputs, context):
        """Return owned hidden states trimmed to the real feature frames."""

        raw = self._validate(outputs, self.outputs)
        hidden = raw[self.output_name]
        return np.array(hidden[0, : context.feature_frames], copy=True)


class Audio2MouthStage(RawStage):
    """Sliding-window AU features -> smoothed mouth parameters ``[N,32]``."""

    stage = "audio2mouth"

    def __init__(self, runner, input_name, output_name, feat_name, quants):
        self.input_name, self.output_name, self.feat_name = (
            input_name,
            output_name,
            feat_name,
        )
        super().__init__(
            runner,
            {input_name: ((1, 30, 50, 512), "float32")},
            {
                output_name: ((1, 30, 32), "float32"),
                feat_name: ((1, 30, 256), "float32"),
            },
            quants,
        )

    @attributed(label="preprocess")
    def preprocess(self, au_window):
        """Own one zero-padded finite float32 [1,30,50,512] window."""

        if (
            not isinstance(au_window, np.ndarray)
            or au_window.ndim != 3
            or au_window.shape[1:] != (50, 512)
            or au_window.dtype != np.dtype("float32")
            or not np.isfinite(au_window).all()
        ):
            raise ValueError("au_window must be finite float32 [<=30,50,512]")
        if au_window.shape[0] > A2M_WINDOW:
            raise ValueError(f"au_window longer than {A2M_WINDOW} frames")
        window = np.zeros((1, A2M_WINDOW, 50, 512), np.float32)
        window[0, : au_window.shape[0]] = au_window
        return PreparedInput({self.input_name: window})

    @attributed(label="postprocess")
    def postprocess(self, outputs):
        """Return the [30,32] parameter rows (aux ``feat`` stays unused)."""

        raw = self._validate(outputs, self.outputs)
        return np.array(raw[self.output_name][0], copy=True)

    def predict(self, au_data):
        """Window loop with fragment crossfade and Butterworth smoothing.

        ``au_data`` is ``[frame_count,50,512]`` (one row per output frame at
        30 fps).  Each call owns one nominal 1-second window (``A2M_WINDOW``
        = 30 rows); consecutive windows normally advance 0.4 s = 12 frames,
        because the next start rewinds ``A2M_CROSSFADE / 10`` seconds from
        the previous end.  The trailing fragment rows are re-anchored
        against the next window with a linear crossfade, and the final
        window slides back so it ends at ``audio_length`` and emits through
        the full window.  The assembled parameter track is low-pass
        filtered per dimension.  Equivalent to calling this stage's
        preprocess/infer/postprocess per window.
        """

        if (
            not isinstance(au_data, np.ndarray)
            or au_data.ndim != 3
            or au_data.shape[1:] != (50, 512)
            or au_data.dtype != np.dtype("float32")
            or not np.isfinite(au_data).all()
        ):
            raise ValueError("au_data must be finite float32 [frame_count,50,512]")
        frame_cnt = au_data.shape[0]
        audio_length = frame_cnt / float(FRAME_RATE)
        interval = 1.0
        frag = A2M_CROSSFADE
        win = A2M_WINDOW
        param_res: list[np.ndarray] = []
        start_time, end_time, is_end = 0.0, interval, False
        while True:
            if end_time >= audio_length:
                is_end = True
                start_time = max(audio_length - interval, 0.0)
                end_time = audio_length
            start_frame = int(start_time * FRAME_RATE)
            end_frame = start_frame + win
            prepared = self.preprocess(au_data[start_frame:end_frame])
            outputs = self.infer(prepared.tensors)
            out = self.postprocess(outputs)
            if start_time == 0.0:
                rng = range(win) if is_end else range(win - frag)
                for tt in rng:
                    param_res.append(out[tt].copy())
            else:
                tail = win if is_end else win - frag
                for tt in range(frag, tail):
                    fid = start_frame + tt
                    if fid < len(param_res):
                        s = min((len(param_res) - fid) / frag, 1.0)
                        param_res[fid] = (1 - s) * out[tt] + s * param_res[fid]
                    else:
                        param_res.append(out[tt].copy())
            start_time = end_time - (frag / 10)
            end_time = start_time + interval
            if is_end:
                break
        params = np.stack(param_res, axis=0).astype(np.float32)
        if len(params) >= frame_cnt:
            params = params[:frame_cnt]
        else:
            pad = np.repeat(params[-1:], frame_cnt - len(params), axis=0)
            params = np.concatenate([params, pad], axis=0)
        return _butter_lowpass(params)


def _butter_lowpass(data, order=4, cutoff=10, fs=FRAME_RATE):
    """Zero-phase Butterworth smoothing along the frame axis (scipy)."""

    from scipy.signal import butter, filtfilt

    wn = 2 * cutoff / fs
    b, a = butter(order, wn, "lowpass", analog=False)
    return filtfilt(b, a, data, padtype=None, axis=0).astype(np.float32)


def interpolate_time_axis(hidden_states, frame_count):
    """Resample ``[T,50,512]`` hidden states to ``[frame_count,50,512]``.

    Uses Torch's linear interpolation with ``align_corners=True`` on the
    time axis, matching the validated source numerics (per layer/dim pair).
    """

    if (
        not isinstance(hidden_states, np.ndarray)
        or hidden_states.ndim != 3
        or hidden_states.shape[1:] != (50, 512)
        or not np.isfinite(hidden_states).all()
    ):
        raise ValueError("hidden_states must be finite [T,50,512]")
    if isinstance(frame_count, bool) or not isinstance(frame_count, Integral):
        raise ValueError("frame_count must be an integer")
    if not 1 <= frame_count <= 100_000:
        raise ValueError("frame_count must be in [1,100000]")
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(hidden_states)).float().permute(1, 2, 0)
    t = F.interpolate(t, size=int(frame_count), mode="linear", align_corners=True)
    return t.permute(2, 0, 1).numpy().astype(np.float32)


class FaceGenEncoderStage(RawStage):
    """Normalized reference image ``[1,3,384,384]`` -> feature pyramid."""

    stage = "facegen_encoder"

    def __init__(self, runner, input_name, f_names, quants):
        self.input_name = input_name
        self.f_names = tuple(f_names)
        shapes = {
            "f0": (1, 3, 384, 384),
            "f1": (1, 16, 192, 192),
            "f2": (1, 32, 96, 96),
            "f3": (1, 64, 48, 48),
        }
        super().__init__(
            runner,
            {input_name: ((1, 3, 384, 384), "float32")},
            {name: (shapes[f"f{index}"], "float32") for index, name in enumerate(self.f_names)},
            quants,
        )

    @attributed(label="preprocess")
    def preprocess(self, image):
        """Own a finite float32 [1,3,384,384] image in [-1,1] RGB layout."""

        return PreparedInput(
            {self.input_name: tensor(image, (1, 3, 384, 384), "reference image")}
        )

    @attributed(label="postprocess")
    def postprocess(self, outputs):
        """Return the owned pyramid tuple (f0, f1, f2, f3).

        Arrays are copied: raw runner outputs may be borrowed SDK buffers,
        and the renderer caches pyramids across the whole clip.
        """

        raw = self._validate(outputs, self.outputs)
        return tuple(np.array(raw[name], copy=True) for name in self.f_names)


class FaceGenDecoderStage(RawStage):
    """Pyramid + ``mouth_param`` [1,32] -> mouth image ``[3,384,384]``."""

    stage = "facegen_decoder"

    def __init__(self, runner, f_names, mouth_name, output_name, quants):
        self.f_names, self.mouth_name, self.output_name = (
            tuple(f_names),
            mouth_name,
            output_name,
        )
        shapes = {
            "f0": (1, 3, 384, 384),
            "f1": (1, 16, 192, 192),
            "f2": (1, 32, 96, 96),
            "f3": (1, 64, 48, 48),
        }
        inputs = {
            name: (shapes[f"f{index}"], "float32")
            for index, name in enumerate(self.f_names)
        }
        inputs[mouth_name] = ((1, 32), "float32")
        super().__init__(
            runner,
            inputs,
            {output_name: ((1, 3, 384, 384), "float32")},
            quants,
        )

    @attributed(label="preprocess")
    def preprocess(self, pyramid, mouth_param):
        """Own the decoder feed: f0/f1/f2/f3 pyramids plus one [1,32] param."""

        if not isinstance(pyramid, tuple) or len(pyramid) != 4:
            raise ValueError("pyramid must be the (f0,f1,f2,f3) tuple")
        values = {name: array for name, array in zip(self.f_names, pyramid)}
        values[self.mouth_name] = mouth_param
        return PreparedInput(self._validate(values, self.inputs))

    @attributed(label="postprocess")
    def postprocess(self, outputs):
        """Return the owned mouth image [3,384,384] in [-1,1] RGB layout."""

        raw = self._validate(outputs, self.outputs)
        return np.array(raw[self.output_name][0], copy=True)


# ======================================================================
# LZ4-compressed background cache (bounded memory on the S100).
# ======================================================================


class CompressedBackground:
    """One background frame held as an LZ4 payload, decompressed on demand.

    The raw ring of 1920x890 BGR frames is ~733 MiB for a 150-frame
    background; LZ4 keeps the resident set ~5x smaller while decompression
    stays far below one frame interval.
    """

    library = None

    def __init__(self, frame):
        if self.library is None:
            path = ctypes.util.find_library("lz4")
            if path is None:
                raise RuntimeError(
                    "Background caching requires the system liblz4 library"
                )
            lib = ctypes.CDLL(path)
            lib.LZ4_compressBound.argtypes = [ctypes.c_int]
            lib.LZ4_compressBound.restype = ctypes.c_int
            lib.LZ4_compress_default.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.LZ4_compress_default.restype = ctypes.c_int
            lib.LZ4_decompress_safe.argtypes = [
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.LZ4_decompress_safe.restype = ctypes.c_int
            type(self).library = lib
        frame = np.ascontiguousarray(frame)
        self.shape, self.dtype, self.nbytes = frame.shape, frame.dtype, frame.nbytes
        buffer = ctypes.create_string_buffer(self.library.LZ4_compressBound(self.nbytes))
        count = self.library.LZ4_compress_default(
            frame.ctypes.data, buffer, self.nbytes, len(buffer)
        )
        if count <= 0:
            raise RuntimeError("Background compression failed")
        self.payload = buffer.raw[:count]
        self.storage_nbytes = count

    @property
    def frame_shape(self):
        """Shape of the decompressed frame (the renderer's geometry)."""

        return self.shape

    def copy(self, out=None):
        """Return a decompressed owned frame, or fill a compatible buffer."""

        frame = np.empty(self.shape, dtype=self.dtype) if out is None else out
        if (
            frame.shape != self.shape
            or frame.dtype != self.dtype
            or not frame.flags.c_contiguous
        ):
            raise ValueError(
                "Background output buffer has an incompatible shape, dtype or layout"
            )
        count = self.library.LZ4_decompress_safe(
            self.payload, frame.ctypes.data, len(self.payload), self.nbytes
        )
        if count != self.nbytes:
            raise RuntimeError("Background decompression failed")
        return frame


# ======================================================================
# Renderer: reference pyramids, feathered mask, per-frame compositing.
# ======================================================================


def resize_bilinear(image, size):
    """Bilinear resize to ``(width, height)`` with OpenCV INTER_LINEAR mapping.

    Same half-pixel-center convention (``src = (dst + 0.5) * scale - 0.5``
    clamped to the source extent), separable per-axis blend of the two
    neighbouring samples, round-to-nearest.  Measured against
    ``cv2.resize(..., interpolation=cv2.INTER_LINEAR)``: maximum deviation
    1 (per-pixel rounding), across square, non-square and the sample's real
    geometries (890x1920 -> 384x384 reference encode and the 384 -> 419x289
    mouth ROI).
    """

    out_h, out_w = size[1], size[0]
    src_h, src_w = image.shape[:2]
    values = image.astype(np.float64)

    def coords(out_n, src_n):
        pos = (np.arange(out_n, dtype=np.float64) + 0.5) * src_n / out_n - 0.5
        pos = np.clip(pos, 0.0, float(src_n - 1))
        low = np.floor(pos).astype(np.int64)
        low = np.minimum(low, src_n - 2 if src_n > 1 else 0)
        return low, pos - low

    x0, xw = coords(out_w, src_w)
    y0, yw = coords(out_h, src_h)
    # Column weights broadcast on axis 1, row weights on axis 0.
    xw_c = xw.reshape((1, out_w) + (1,) * (image.ndim - 2))
    yw_c = yw.reshape((out_h,) + (1,) * (image.ndim - 1))
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)

    def horiz(rows):
        return rows[:, x0] * (1.0 - xw_c) + rows[:, x1] * xw_c

    blended = horiz(values[y0]) * (1.0 - yw_c) + horiz(values[y1]) * yw_c
    return np.rint(blended).astype(np.uint8)


def load_reference_bgr(path):
    """Read a reference JPEG as BGR uint8 via Pillow (no OpenCV).

    Pillow and OpenCV decode the sample's reference JPEGs bit-identically
    (measured over all 150 files).
    """

    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def normalize_reference(image_bgr):
    """BGR uint8 image -> finite float32 [1,3,384,384] in [-1,1] RGB."""

    img = image_bgr[:, :, 0:3][:, :, ::-1]
    img = resize_bilinear(np.ascontiguousarray(img), (384, 384))
    t = img.astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))
    t = 2.0 * t - 1.0
    return t[None, ...].astype(np.float32)


def background_index_for(index, count):
    """Ping-pong background ring order: forward, then backward, then repeat."""

    if (index // count) % 2 == 0:
        return index % count
    return count - 1 - (index % count)


class AvatarRenderer:
    """Cache reference pyramids and composite mouth frames over backgrounds.

    Memory contract: the ring stores each frame as a compressed payload;
    the raw decoded list is dropped as soon as compression holds the pixels,
    before the reference pyramids load (injecting frames through
    ``background_frames`` keeps the caller's list untouched).

    Attributes:
        encoder_stage: FaceGenEncoderStage used once per reference frame.
        decoder_stage: FaceGenDecoderStage used once per rendered frame.
        bg: LZ4-compressed background frames (owned payload bytes).
        ref_feats: per-reference (f0,f1,f2,f3) pyramid tuples.
        y1/y2/x1/x2: face box in background-frame pixel coordinates.
    """

    def __init__(self, data_dir, encoder_stage, decoder_stage, max_refs=150,
                 *, background_frames=None):
        from scipy import ndimage

        self.encoder_stage = encoder_stage
        self.decoder_stage = decoder_stage
        data_dir = Path(data_dir)
        injected = background_frames is not None
        if background_frames is None:
            # Board path: hardware (VPU) decode of the prepared H.264
            # background; a legacy MPEG-4 asset raises with the host
            # preparation command instead of any software fallback.
            from samples.vision.lite_avatar.runtime.python.media_worker import (
                BackgroundFormatError,
                decode_background,
            )

            try:
                background_frames = decode_background(data_dir, max_refs)
            except BackgroundFormatError as error:
                raise ValueError(str(error)) from error
        self.bg = []
        for index, img in enumerate(background_frames):
            if img is None or img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 1:
                raise ValueError(f"Invalid background frame {index}: {getattr(img, 'shape', None)}")
            if self.bg and img.shape != self.bg[0].frame_shape:
                raise ValueError(
                    f"Background frame {index} is {img.shape[:2]} but "
                    f"frame 0 was {self.bg[0].frame_shape[:2]}; one geometry is required"
                )
            self.bg.append(CompressedBackground(img))
        # The ring never cycles past max_refs slots; frames beyond that are
        # not cached, no matter how long the source loop is.
        self.bg = self.bg[:max_refs]
        if not injected:
            # The compressed ring owns the pixels now; the raw decoded list
            # (~733 MiB for 150 890x1920 BGR frames) is dead weight for the
            # reference-pyramid pass below, so release it before the loads.
            # Caller-injected frames are never touched: the caller owns them.
            background_frames = None
        if not self.bg:
            raise ValueError(f"No background frames from {data_dir}")
        height, width = self.bg[0].frame_shape[:2]
        first = (data_dir / "face_box.txt").read_text(encoding="utf-8").splitlines()[0]
        box = first.split()
        if len(box) != 4:
            raise ValueError("face_box.txt must hold exactly y1 y2 x1 x2")
        try:
            self.y1, self.y2, self.x1, self.x2 = (int(value) for value in box)
        except ValueError as error:
            raise ValueError(f"face_box.txt holds non-integer values: {box}") from error
        if not (0 <= self.y1 < self.y2 <= height and 0 <= self.x1 < self.x2 <= width):
            raise ValueError(
                f"face box y[{self.y1},{self.y2}) x[{self.x1},{self.x2}) lies "
                f"outside the {width}x{height} background"
            )
        if self.y2 - self.y1 <= 20 or self.x2 - self.x1 <= 20:
            raise ValueError(
                "face box ROI must be larger than the 10 px feather band on "
                "every side (needs > 20 px in both axes)"
            )
        mask = (np.ones((self.y2 - self.y1, self.x2 - self.x1, 3)) * 255).astype(np.uint8)
        mask[10:-10, 10:-10, :] *= 0
        # Feathered blend band: gaussian sigma 15 over the 21x21 kernel
        # support with edge-reflection (cv2 BORDER_REFLECT_101 semantics,
        # scipy "mirror"); measured max deviation 0.0043 vs cv2.GaussianBlur.
        blurred = ndimage.gaussian_filter(
            mask.astype(np.float32) / 255.0, 15, mode="mirror", truncate=10 / 15
        )
        self.mask = blurred.astype(np.float32)
        self.inverse_mask = 1 - self.mask
        # Ring slots pair 1:1 with reference pyramids: the ring index selects
        # both the background frame and its reference features, so exactly
        # bg_n pyramids are encoded — never the whole reference directory.
        self.bg_n = min(max_refs, len(self.bg))
        self.ref_feats: list[tuple] = []
        for ii in range(self.bg_n):
            path = data_dir / "ref_frames" / f"ref_{ii:05d}.jpg"
            if not path.is_file():
                break
            try:
                image = load_reference_bgr(path)
            except OSError as error:
                raise ValueError(
                    f"Could not decode reference frame {path}: {error}"
                ) from error
            prepared = self.encoder_stage.preprocess(normalize_reference(image))
            outputs = self.encoder_stage.infer(prepared.tensors)
            self.ref_feats.append(self.encoder_stage.postprocess(outputs))
        if not self.ref_feats:
            raise ValueError(f"No reference frames under {data_dir / 'ref_frames'}")
        if len(self.ref_feats) < self.bg_n:
            raise ValueError(
                f"The background ring cycles {self.bg_n} frame slots but only "
                f"{len(self.ref_feats)} reference frames decoded under "
                f"{data_dir / 'ref_frames'}; provide at least one reference "
                "frame per ring slot"
            )

    @property
    def width(self):
        return int(self.bg[0].shape[1])

    @property
    def height(self):
        return int(self.bg[0].shape[0])

    @property
    def cache_mib(self):
        return sum(item.storage_nbytes for item in self.bg) / 2**20

    def gen_frame(self, param, bg_id, out=None):
        """Render one owned BGR frame (or fill ``out``) for a mouth param.

        Compositing order is fixed: decoder mouth image -> ``*0.5+0.5`` ->
        clip [0,1] -> ``*255`` -> uint8 -> HWC -> ROI resize -> RGB->BGR ->
        feathered blend over the decompressed background.
        """

        prepared = self.decoder_stage.preprocess(self.ref_feats[bg_id], param)
        outputs = self.decoder_stage.infer(prepared.tensors)
        mouth = self.decoder_stage.postprocess(outputs)
        np.multiply(mouth, 0.5, out=mouth)
        np.add(mouth, 0.5, out=mouth)
        np.clip(mouth, 0, 1, out=mouth)
        np.multiply(mouth, 255, out=mouth)
        mouth = np.transpose(mouth, (1, 2, 0)).astype(np.uint8)
        mouth = resize_bilinear(mouth, (self.x2 - self.x1, self.y2 - self.y1))
        mouth = mouth[:, :, ::-1]
        full = self.bg[bg_id].copy(out=out)
        roi = full[self.y1 : self.y2, self.x1 : self.x2, :]
        full[self.y1 : self.y2, self.x1 : self.x2, :] = (
            mouth * self.inverse_mask + roi * self.mask
        )
        return full


# ======================================================================
# Application-level four-model composition.
# ======================================================================


@dataclass(frozen=True)
class AvatarResult:
    """Rendered-clip metadata plus a bounded owned-frame iterator.

    Attributes:
        frame_count: frames at 30 fps (== parameter rows).
        width/height: background-frame geometry in pixels.
        feature_frames: real (unpadded) LFR feature frames of the audio.
        renderer: the caching renderer; shared with the pipeline.
        params: mouth parameters ``[frame_count,32]`` float32.
    """

    frame_count: int
    width: int
    height: int
    feature_frames: int
    renderer: AvatarRenderer
    params: np.ndarray

    def frames(self):
        """Yield ``frame_count`` owned BGR frames in presentation order."""

        for index in range(self.frame_count):
            yield self.render_frame(index)

    def render_frame(self, index, out=None):
        """Render presentation frame ``index`` (owned, or into ``out``)."""

        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("index must be an integer")
        if index < 0:
            raise ValueError("index must be nonnegative")
        param = self.params[index % self.frame_count]
        return self.renderer.gen_frame(
            param[None, :], background_index_for(index, self.renderer.bg_n), out=out
        )


class LiteAvatarPipeline:
    """Compose audio encoder -> audio2mouth -> face-generator rendering.

    Attributes:
        runners: Ordered audio_encoder, audio2mouth, facegen_encoder and
            facegen_decoder lazy runners (loaded NamedArrayRunner instances).
        names: Bound physical tensor names for the complete model group.
        frontend: The deterministic LFR/CMVN feature frontend.
        data_dir: Avatar data directory (prepared bg_video.h264 with its
            bg_video.json sidecar, face_box.txt, ref_frames/).
        audio_released: True once the audio models were freed after the
            last ``predict`` audio phase (they reload on the next call).
        last_timings: Seconds of the most recent predict's phases:
            ``audio`` (features+encoder+a2m) and ``renderer_init``.
    """

    def __init__(
        self, frontend, runners, names, data_dir, output_quants, *,
        background_frames=None
    ):
        if len(runners) != len(STAGES) or not all(callable(runner) for runner in runners):
            raise TypeError("Expected four callable model runners")
        if not isinstance(names, AvatarTensorNames):
            raise TypeError("Expected AvatarTensorNames from model binding")
        if not isinstance(frontend, AvatarFrontend):
            raise TypeError("Expected an AvatarFrontend")
        if len(tuple(output_quants)) != len(STAGES):
            raise TypeError("Expected output-quant snapshots for all four stages")
        self.frontend = frontend
        self.runners = tuple(runners)
        self.names = names
        self.data_dir = Path(data_dir)
        # Host tests inject the decoded background frames here; on the board
        # the renderer loads them from the prepared asset (VPU decode).
        self._background_frames = background_frames
        self.audio_released = False
        self.last_timings: dict[str, float] = {}
        self._renderer: AvatarRenderer | None = None
        self._schedule_request: dict | None = None
        quants = tuple(output_quants)
        self.audio_encoder_stage = AudioEncoderStage(
            lambda feed: self._run_stage(0, feed),
            names.audio_encoder_input,
            names.audio_encoder_output,
            quants[0],
        )
        self.audio2mouth_stage = Audio2MouthStage(
            lambda feed: self._run_stage(1, feed),
            names.audio2mouth_input,
            names.audio2mouth_output,
            names.audio2mouth_feat,
            quants[1],
        )
        self.facegen_encoder_stage = FaceGenEncoderStage(
            lambda feed: self._run_stage(2, feed),
            names.facegen_encoder_input,
            (
                names.facegen_encoder_f0,
                names.facegen_encoder_f1,
                names.facegen_encoder_f2,
                names.facegen_encoder_f3,
            ),
            quants[2],
        )
        self.facegen_decoder_stage = FaceGenDecoderStage(
            lambda feed: self._run_stage(3, feed),
            (
                names.facegen_decoder_f0,
                names.facegen_decoder_f1,
                names.facegen_decoder_f2,
                names.facegen_decoder_f3,
            ),
            names.facegen_decoder_mouth,
            names.facegen_decoder_output,
            quants[3],
        )

    def _run_stage(self, index, tensors):
        """Call one runner, restoring scheduling before a fresh load's first run.

        A runner that was unloaded (audio release, renderer-init handoff)
        rebuilds its SDK runtime on the next call.  Scheduling parameters are
        a property of the loaded runtime — and ``set_scheduling_params``
        performs the load itself — so the stored request is applied before
        the first inference of every fresh load: no reload can spend even
        one frame without the requested priority/cores.
        """

        runner = self.runners[index]
        if not runner.loaded and self._schedule_request is not None:
            runner.set_scheduling_params(**self._schedule_request)
        return runner(tensors)

    @classmethod
    def from_models(cls, selections, cmvn_path, data_dir, *, runtime_factory=None):
        """Load and bind the four models for inference on the board.

        Args:
            selections: Ordered audio_encoder, audio2mouth, facegen_encoder
                and facegen_decoder selections for one published target
                (s100, s100p or s600).
            cmvn_path: Pinned ``am.mvn`` for the feature frontend.
            data_dir: Avatar data directory (background video, face box and
                reference frames).
            runtime_factory: Optional SDK factory used for injected runtimes.

        Returns:
            LiteAvatarPipeline: A pipeline ready for predict.

        Raises:
            ValueError: Selections or tensor metadata do not match the
                declared four-model contract.
            RuntimeError: SDK construction or model loading fails.
        """

        runners, names, output_quants = load_model_runners(
            selections, runtime_factory=runtime_factory
        )
        frontend = AvatarFrontend(cmvn_path)
        return cls(frontend, runners, names, data_dir, output_quants)

    def set_scheduling_params(self, *, priority=None, bpu_cores=None):
        """Store and apply model-keyed scheduling to the loaded runtimes.

        A partial call updates only the values it names; earlier non-None
        settings survive.  Currently loaded runtimes receive the request
        immediately; runners that are unloaded (audio models after a
        release, face-generator models before renderer init) receive it
        before their next inference, so the request survives the pipeline's
        load/unload cycles.
        """

        if priority is None and bpu_cores is None:
            return
        if priority is not None and not 0 <= priority <= 255:
            raise ValueError("priority must be in [0,255]")
        if bpu_cores is not None and (
            not bpu_cores or any(core < 0 for core in bpu_cores)
        ):
            raise ValueError("bpu_cores must be nonnegative core indexes")
        request = dict(self._schedule_request or {})
        if priority is not None:
            request["priority"] = priority
        if bpu_cores is not None:
            request["bpu_cores"] = bpu_cores
        self._schedule_request = request
        for runner in self.runners:
            if runner.loaded:
                runner.set_scheduling_params(**request)

    def release_audio_models(self):
        """Unload the audio encoder/audio2mouth runtimes before renderer work.

        The renderer's reference pyramids are ~700 MiB raw; the audio models
        (~257 MiB of artifacts plus workspaces) are dead weight once mouth
        parameters exist.  A later ``predict`` reloads them lazily (and
        replays the stored scheduling request); unloading an already
        unloaded runner is a no-op, so the release is idempotent.
        """

        self.runners[0].unload()
        self.runners[1].unload()
        self.audio_released = True
        gc.collect()

    def _ensure_renderer(self):
        if self._renderer is None:
            self._renderer = AvatarRenderer(
                self.data_dir,
                self.facegen_encoder_stage,
                self.facegen_decoder_stage,
                background_frames=self._background_frames,
            )
        return self._renderer

    def predict(self, avatar_input: AvatarInput) -> AvatarResult:
        """Render one clip: audio features to owned BGR frames.

        The two audio stages run first and their models are released before
        the renderer loads the face-generator models and caches reference
        pyramids (S100 memory discipline).  Returns once parameters exist;
        frames are produced lazily by the result.
        """

        from time import perf_counter

        if not isinstance(avatar_input, AvatarInput):
            raise ValueError("Expected an AvatarInput")
        audio = prepare_waveform(avatar_input.waveform, avatar_input.sample_rate)
        frame_count = int(len(audio) * FRAME_RATE // 16000)
        if frame_count < 1:
            raise ValueError("Audio shorter than one output frame (1/30 s at 16 kHz)")
        start = perf_counter()
        # The audio models become resident again for this call; the flag is
        # reset up front so it stays truthful even if a stage below raises.
        self.audio_released = False
        feats = self.frontend.features_from_mono(audio)
        prepared = self.audio_encoder_stage.preprocess(feats)
        context = AvatarContext(prepared.context.feature_frames, frame_count)
        raw = self.audio_encoder_stage.infer(prepared.tensors)
        hidden = self.audio_encoder_stage.postprocess(raw, context)
        au = interpolate_time_axis(hidden, frame_count)
        params = self.audio2mouth_stage.predict(au)
        audio_seconds = perf_counter() - start
        self.release_audio_models()
        start = perf_counter()
        renderer = self._ensure_renderer()
        renderer_seconds = perf_counter() - start
        self.last_timings = {
            "audio": audio_seconds,
            "renderer_init": renderer_seconds,
        }
        return AvatarResult(
            frame_count,
            renderer.width,
            renderer.height,
            context.feature_frames,
            renderer,
            params,
        )


# ======================================================================
# Four lazy raw runners with explicit model binding.
# ======================================================================


@dataclass(frozen=True)
class AvatarTensorNames:
    """Physical tensor names supplied by model binding (no positional use)."""

    audio_encoder_input: str
    audio_encoder_output: str
    audio2mouth_input: str
    audio2mouth_output: str
    audio2mouth_feat: str
    facegen_encoder_input: str
    facegen_encoder_f0: str
    facegen_encoder_f1: str
    facegen_encoder_f2: str
    facegen_encoder_f3: str
    facegen_decoder_f0: str
    facegen_decoder_f1: str
    facegen_decoder_f2: str
    facegen_decoder_f3: str
    facegen_decoder_mouth: str
    facegen_decoder_output: str

    def __post_init__(self):
        from dataclasses import fields

        if any(
            not isinstance(getattr(self, field.name), str) or not getattr(self, field.name)
            for field in fields(self)
        ):
            raise ValueError("Every tensor role requires a nonempty name")


def load_model_runners(selections, *, runtime_factory=None):
    """Validate and load the four physical model contracts.

    All four bindings are validated up front (fail fast on a wrong or
    rebuilt artifact), then the two face-generator runtimes are unloaded so
    only the audio pair stays resident while features are computed; the
    renderer reloads them lazily after the audio models are released.

    Args:
        selections: Ordered stage selections for the four models.
        runtime_factory: Optional injected SDK factory accepting a path.

    Returns:
        tuple: Four NamedArrayRunner instances, the bound
            :class:`AvatarTensorNames`, and the per-stage output-quant
            descriptor snapshots captured while every binding was loaded
            (the facegen runners are returned unloaded and rebind lazily,
            so the pipeline must not re-read their bindings at init).

    Raises:
        ValueError: A selection or tensor contract is invalid.
        RuntimeError: Board SDK loading fails. Normal loading checks board
            identity and asset files before constructing the SDK.
    """

    if tuple(selection.stage for selection in selections) != STAGES:
        raise ValueError("Expected ordered four LiteAvatar stage selections")
    from samples.vision.lite_avatar.runtime.python.cli import PUBLISHED_TARGETS

    targets = {selection.target for selection in selections}
    if len(targets) != 1:
        raise ValueError(
            "LiteAvatar stages must come from one target's four-model set "
            f"(got {sorted(targets)})"
        )
    if targets.pop() not in PUBLISHED_TARGETS:
        raise ValueError(
            "LiteAvatar publishes the four-model set for "
            f"{', '.join(PUBLISHED_TARGETS)}"
        )
    for selection in selections:
        validate_selection(selection)
    runners = tuple(
        NamedArrayRunner(
            selection,
            binding_loader=bind_model,
            physical_inputs=physical_inputs,
            task_name=f"LiteAvatar {selection.stage}",
            runtime_factory=runtime_factory,
        )
        for selection in selections
    )
    bindings = [runner.load() for runner in runners]
    # Snapshot the output-quant facts while every binding is loaded, then
    # detach them: the two facegen runtimes unload below, and their SDK
    # descriptors must not ride along into long-lived stage state.
    output_quants = tuple(
        detach_quants(binding.metadata.output_quants) for binding in bindings
    )
    names = AvatarTensorNames(
        audio_encoder_input=bindings[0].inputs["feats"],
        audio_encoder_output=bindings[0].outputs["hidden_states"],
        audio2mouth_input=bindings[1].inputs["input_au"],
        audio2mouth_output=bindings[1].outputs["output"],
        audio2mouth_feat=bindings[1].outputs["feat"],
        facegen_encoder_input=bindings[2].inputs["image"],
        facegen_encoder_f0=bindings[2].outputs["f0"],
        facegen_encoder_f1=bindings[2].outputs["f1"],
        facegen_encoder_f2=bindings[2].outputs["f2"],
        facegen_encoder_f3=bindings[2].outputs["f3"],
        facegen_decoder_f0=bindings[3].inputs["f0"],
        facegen_decoder_f1=bindings[3].inputs["f1"],
        facegen_decoder_f2=bindings[3].inputs["f2"],
        facegen_decoder_f3=bindings[3].inputs["f3"],
        facegen_decoder_mouth=bindings[3].inputs["mouth_param"],
        facegen_decoder_output=bindings[3].outputs["mouth_image"],
    )
    runners[2].unload()
    runners[3].unload()
    return runners, names, output_quants


__all__ = [
    "A2M_CROSSFADE",
    "A2M_WINDOW",
    "AvatarContext",
    "AvatarInput",
    "AvatarRenderer",
    "AvatarResult",
    "AvatarTensorNames",
    "Binding",
    "CompressedBackground",
    "FRAME_RATE",
    "INPUTS",
    "LiteAvatarPipeline",
    "MAX_FEATURE_FRAMES",
    "OUTPUTS",
    "OwnedQuantDescriptor",
    "PreparedInput",
    "RawStage",
    "StageError",
    "STAGES",
    "background_index_for",
    "bind_model",
    "detach_quants",
    "float_outputs",
    "interpolate_time_axis",
    "load_model_runners",
    "normalize_reference",
    "physical_inputs",
    "validate_selection",
]
