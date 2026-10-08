"""Pinned FunASR CPU feature preparation; audio file I/O stays outside."""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from threading import RLock

import numpy as np

from utils.py_utils.assets import sha256_file
from samples.speech.paraformer.runtime.python.model_binding import LOCAL_DIGESTS

_RNG_LOCK = RLock()


@dataclass(frozen=True)
class PreparedFeatures:
    tensor: np.ndarray
    valid_frames: int
    original_frames: int
    truncated: bool
    sample_count: int


def prepare_waveform(waveform, sample_rate):
    """Return owned mono float32 samples, preserving source arithmetic."""
    if (
        isinstance(sample_rate, (bool, np.bool_))
        or not isinstance(sample_rate, Integral)
        or sample_rate != 16000
    ):
        raise ValueError("Paraformer requires 16000 Hz audio; no implicit resampling")
    if (
        not isinstance(waveform, np.ndarray)
        or waveform.dtype != np.float32
        or waveform.ndim not in (1, 2)
        or waveform.shape[0] == 0
        or (waveform.ndim == 2 and waveform.shape[1] == 0)
        or not np.isfinite(waveform).all()
    ):
        raise ValueError(
            "Expected finite float32 mono/multichannel audio with at least one sample"
        )
    mono = waveform.mean(axis=1) if waveform.ndim == 2 else waveform
    if not np.isfinite(mono).all():
        raise ValueError("Channel averaging produced non-finite samples")
    return np.array(mono, dtype=np.float32, order="C", copy=True)


class ParaformerFrontend:
    """Source-compatible 80-bin fbank, 7/6 LFR and pinned CMVN on CPU."""

    def __init__(self, cmvn_path, *, random_seed=191009):
        self.cmvn_path = Path(cmvn_path).expanduser()
        if (
            not self.cmvn_path.is_file()
            or sha256_file(self.cmvn_path) != LOCAL_DIGESTS["am.mvn"]
        ):
            raise ValueError("CMVN must match the pinned Paraformer am.mvn")
        if type(random_seed) is not int or not 0 <= random_seed < 2**63:
            raise ValueError("random_seed must be an integer in [0,2**63)")
        self.random_seed = random_seed
        try:
            import torch
            from funasr.frontends.wav_frontend import WavFrontend
        except ImportError as error:
            raise RuntimeError(
                "Audio preprocessing requires Torch, torchaudio and FunASR; see the Python README"
            ) from error
        self._torch = torch
        self._frontend = WavFrontend(
            cmvn_file=str(self.cmvn_path),
            fs=16000,
            window="hamming",
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            lfr_m=7,
            lfr_n=6,
        )

    def pre_process(self, waveform, sample_rate):
        """Return fixed [1,400,560] features and explicit truncation metadata.

        Scope CPU RNG changes to this call, including exception paths. The lock
        serializes this adapter's calls; unrelated threads using Torch's global
        RNG must still be coordinated by the application.
        """
        mono = prepare_waveform(waveform, sample_rate)
        torch = self._torch
        tensor = torch.from_numpy(mono).unsqueeze(0)
        with _RNG_LOCK, torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.random.default_generator.manual_seed(self.random_seed)
            features, lengths = self._frontend(
                tensor, torch.tensor([len(mono)], dtype=torch.int64)
            )
        if (
            features.ndim != 3
            or features.shape[0] != 1
            or features.shape[2] != 560
            or lengths.numel() != 1
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("FunASR returned incompatible feature geometry or lengths")
        original_frames = int(lengths[0])
        if not 0 < original_frames <= features.shape[1]:
            raise ValueError("FunASR returned an invalid feature-frame count")
        valid_frames = min(original_frames, 400)
        values = features[0, :valid_frames].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("FunASR produced non-finite features")
        output = np.zeros((1, 400, 560), np.float32)
        output[0, :valid_frames] = values
        return PreparedFeatures(
            output, valid_frames, original_frames, original_frames > 400, len(mono)
        )
