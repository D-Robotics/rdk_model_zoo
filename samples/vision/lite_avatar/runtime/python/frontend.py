"""Deterministic audio frontend for LiteAvatar: waveform -> [T, 560] features.

The feature chain is the one the upstream HumanAIGC/lite-avatar project feeds to
its Paraformer-derived audio encoder (torchaudio Kaldi fbank -> 7/6 low-frame-
rate stacking -> CMVN).  The arithmetic here is a faithful port of the vendored
``funasr_local`` ``WavFrontend`` path (MIT, Copyright (c) Alibaba, Inc. and its
affiliates; parts borrowed from espnet/espnet), with the vendored package
itself replaced by this single self-contained module.

Three policies differ from the Model Zoo Paraformer sample's frontend on
purpose and must not be "unified": LiteAvatar's source computes with
``dither=0.0`` (deterministic features), keeps the **first** channel of stereo
input, and resamples non-16 kHz audio with polyphase filtering instead of
rejecting it.  The Paraformer frontend (seeded dither, channel mean, 16 kHz
only) is untouched by this sample.

Only the fbank itself needs Torch; the CMVN parsing, LFR stacking and CMVN
application below are plain NumPy so they are unit-testable on a host without
Torch installed.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np

from utils.py_utils.assets import sha256_file
from samples.vision.lite_avatar.runtime.python.cli import LOCAL_DIGESTS

#: Sample rate the audio encoder expects, in Hz.
SAMPLE_RATE = 16000
#: Kaldi fbank parameters (identical to the upstream LiteAvatar constructor).
N_MELS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
WINDOW = "hamming"
DITHER = 0.0
#: Low-frame-rate stacking: concatenate ``LFR_M`` frames with stride ``LFR_N``.
LFR_M = 7
LFR_N = 6
#: Feature dimension after LFR stacking (80 mels x 7 frames).
FEATURE_DIM = N_MELS * LFR_M
#: Shortest window (in samples) that yields one fbank frame with snip_edges.
MIN_SAMPLES = SAMPLE_RATE * FRAME_LENGTH_MS // 1000


def load_cmvn(cmvn_path) -> np.ndarray:
    """Parse a Kaldi text CMVN file into a ``[2, dim]`` float32 array.

    Row 0 holds the ``<AddShift>`` means and row 1 the ``<Rescale>``
    (variance) coefficients, exactly as the upstream ``load_cmvn`` reads
    them (the trailing ``</AddShift>``/``</Rescale>`` token is dropped by
    the ``[3:-1]`` slice).
    """

    path = Path(cmvn_path).expanduser()
    lines = path.read_text(encoding="utf-8").splitlines()
    means: list[str] = []
    variances: list[str] = []
    for i, line in enumerate(lines):
        items = line.split()
        if not items:
            continue
        if items[0] == "<AddShift>":
            following = lines[i + 1].split()
            if following[0] == "<LearnRateCoef>":
                means = following[3 : len(following) - 1]
        elif items[0] == "<Rescale>":
            following = lines[i + 1].split()
            if following[0] == "<LearnRateCoef>":
                variances = following[3 : len(following) - 1]
    if not means or not variances or len(means) != len(variances):
        raise ValueError(f"CMVN file is missing AddShift/Rescale rows: {path}")
    cmvn = np.asarray([means, variances], dtype=np.float32)
    if not np.isfinite(cmvn).all():
        raise ValueError(f"CMVN file contains non-finite coefficients: {path}")
    return cmvn


def apply_lfr(inputs: np.ndarray, lfr_m: int = LFR_M, lfr_n: int = LFR_N) -> np.ndarray:
    """Stack frames into LFR windows: ``[T, mel] -> [ceil(T/lfr_n), mel*lfr_m]``.

    Mirrors the upstream ``apply_lfr``: the sequence is left-padded with
    ``(lfr_m - 1) // 2`` copies of the first frame, windows of ``lfr_m``
    frames advance by ``lfr_n``, and a truncated final window is padded by
    repeating the last frame's coefficients.
    """

    if inputs.ndim != 2 or inputs.shape[0] == 0:
        raise ValueError("LFR stacking needs a non-empty [frames, dim] matrix")
    total = inputs.shape[0]
    frames: list[np.ndarray] = []
    left_padding = np.repeat(inputs[0:1], (lfr_m - 1) // 2, axis=0)
    padded = np.vstack((left_padding, inputs))
    total = total + (lfr_m - 1) // 2
    for i in range(int(np.ceil(inputs.shape[0] / lfr_n))):
        start = i * lfr_n
        if lfr_m <= total - start:
            frames.append(padded[start : start + lfr_m].reshape(-1))
        else:  # last window: pad with the final frame's coefficients
            remaining = lfr_m - (total - start)
            frame = padded[start:].reshape(-1)
            for _ in range(remaining):
                frame = np.hstack((frame, padded[-1]))
            frames.append(frame)
    return np.vstack(frames).astype(np.float32, copy=False)


def apply_cmvn(features: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
    """Normalise features in place, preserving the upstream op order.

    Upstream semantics: ``features += means`` **then** ``features *= vars``
    (the ``am.mvn`` coefficients are applied as add/multiply, not
    subtract/divide).
    """

    dim = features.shape[1]
    if cmvn.shape[1] < dim:
        raise ValueError("CMVN dimension is smaller than the feature dimension")
    features = features.astype(np.float32, copy=True)
    features += cmvn[0, :dim]
    features *= cmvn[1, :dim]
    return features


def prepare_waveform(waveform, sample_rate: int) -> np.ndarray:
    """Return owned mono float32 16 kHz samples for the fbank stage.

    Order matches the upstream loader: polyphase resample when the rate is
    not 16 kHz (operating on all channels), then cast to float32, then keep
    the **first** channel.
    """

    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, Integral)
        or int(sample_rate) <= 0
    ):
        raise ValueError("sample_rate must be a positive integer")
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.ndim not in (1, 2)
        or waveform.shape[0] == 0
        or (waveform.ndim == 2 and waveform.shape[1] == 0)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError("Expected finite mono/multichannel samples with at least one sample")
    audio = waveform
    rate = int(sample_rate)
    if rate != SAMPLE_RATE:
        from fractions import Fraction

        from scipy.signal import resample_poly

        ratio = Fraction(SAMPLE_RATE, rate).limit_denominator(10000)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if audio.shape[0] < MIN_SAMPLES:
        raise ValueError(
            f"Audio is too short for one {FRAME_LENGTH_MS} ms frame "
            f"(need >= {MIN_SAMPLES} samples at {SAMPLE_RATE} Hz, got {audio.shape[0]})"
        )
    return np.ascontiguousarray(audio)


class AvatarFrontend:
    """Waveform -> LFR-CMVN fbank features for the LiteAvatar audio encoder."""

    def __init__(self, cmvn_path):
        path = Path(cmvn_path).expanduser()
        if (
            not path.is_file()
            or sha256_file(path) != LOCAL_DIGESTS["am.mvn"]
        ):
            raise ValueError("CMVN must match the pinned LiteAvatar am.mvn")
        self.cmvn = load_cmvn(path)
        if self.cmvn.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected {FEATURE_DIM}-dim CMVN coefficients, got {self.cmvn.shape[1]}"
            )

    def features(self, waveform, sample_rate: int) -> np.ndarray:
        """Compute deterministic ``[T, 560]`` features (T = LFR frame count)."""

        return self.features_from_mono(prepare_waveform(waveform, sample_rate))

    def features_from_mono(self, audio: np.ndarray) -> np.ndarray:
        """Compute features for already-prepared 16 kHz mono float32 samples."""

        if (
            not isinstance(audio, np.ndarray)
            or audio.ndim != 1
            or audio.dtype != np.dtype("float32")
            or audio.shape[0] < MIN_SAMPLES
            or not np.isfinite(audio).all()
        ):
            raise ValueError("Expected prepared 16 kHz mono float32 samples")
        mat = self._fbank(audio)
        stacked = apply_lfr(mat)
        return apply_cmvn(stacked, self.cmvn)

    def _fbank(self, audio: np.ndarray) -> np.ndarray:
        """80-bin hamming fbank with dither disabled (Torch, lazily imported)."""

        try:
            import torch
            from torchaudio.compliance import kaldi
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Audio feature extraction requires Torch and torchaudio; "
                "see the runtime README for the pinned CPU wheels"
            ) from error
        with torch.no_grad():
            waveform = torch.from_numpy(audio).unsqueeze(0)
            waveform = waveform * (1 << 15)
            mat = kaldi.fbank(
                waveform,
                num_mel_bins=N_MELS,
                frame_length=FRAME_LENGTH_MS,
                frame_shift=FRAME_SHIFT_MS,
                dither=DITHER,
                energy_floor=0.0,
                window_type=WINDOW,
                sample_frequency=SAMPLE_RATE,
                snip_edges=True,
            )
        out = mat.numpy()
        if out.ndim != 2 or out.shape[1] != N_MELS or out.shape[0] == 0:
            raise ValueError("fbank returned an unexpected geometry")
        return out.astype(np.float32, copy=False)


__all__ = [
    "AvatarFrontend",
    "DITHER",
    "FEATURE_DIM",
    "FRAME_LENGTH_MS",
    "FRAME_SHIFT_MS",
    "LFR_M",
    "LFR_N",
    "MIN_SAMPLES",
    "N_MELS",
    "SAMPLE_RATE",
    "WINDOW",
    "apply_cmvn",
    "apply_lfr",
    "load_cmvn",
    "prepare_waveform",
]
