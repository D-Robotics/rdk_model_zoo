"""Paraformer S100 HBM inference pipeline.

The public input is a 16 kHz WAV file. FunASR WavFrontend creates the validated
fixed-shape fbank+LFR+CMVN tensor before the preserved deployment sequence:
Encoder HBM, Predictor HBM, CPU CIF, Decoder HBM, then greedy token decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Union
import json

import numpy as np
import soundfile as sound_file
import torch
from funasr.frontends.wav_frontend import WavFrontend
from hbm_runtime import HB_HBMRuntime


@dataclass
class ParaformerFrontendConfig:
    """Configuration matching the upstream Paraformer WavFrontend.

    Attributes:
        cmvn_path: Path to the upstream Paraformer CMVN statistics file.
        sample_rate: Required WAV sample rate in Hz.
        max_frames: Fixed encoder frame limit after LFR processing.
        random_seed: Random seed for FunASR fbank dithering.
    """

    cmvn_path: str = '/opt/hobot/model/s100/basic/paraformer/am.mvn'
    sample_rate: int = 16000
    max_frames: int = 400
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6
    random_seed: int = 191009


class ParaformerFrontend:
    """Convert WAV audio into the fixed-shape Paraformer encoder input."""

    def __init__(self, config: ParaformerFrontendConfig) -> None:
        """Create the upstream-compatible FunASR WavFrontend.

        Args:
            config: CMVN path and frontend parameters from `paraformer_config.yaml`.
        """
        self.config = config
        self.frontend = WavFrontend(
            cmvn_file=config.cmvn_path,
            fs=config.sample_rate,
            window='hamming',
            n_mels=config.n_mels,
            frame_length=config.frame_length,
            frame_shift=config.frame_shift,
            lfr_m=config.lfr_m,
            lfr_n=config.lfr_n,
        )

    def pre_process(self, audio_file: Union[str, Path]) -> tuple[np.ndarray, int]:
        """Load one WAV file and generate fbank, LFR, and CMVN features.

        Args:
            audio_file: Mono-compatible 16 kHz PCM WAV file.

        Returns:
            Fixed float32 `[1, 400, 560]` features and valid LFR frame count.

        Raises:
            ValueError: If the WAV sample rate or content is invalid.
        """
        waveform, sample_rate = sound_file.read(str(audio_file), dtype='float32')
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f'Expected {self.config.sample_rate} Hz WAV, got {sample_rate} Hz: {audio_file}'
            )
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f'Expected non-empty mono-compatible WAV data: {audio_file}')
        waveform_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).float().unsqueeze(0)
        torch.manual_seed(self.config.random_seed)
        features, lengths = self.frontend(
            waveform_tensor,
            torch.tensor([waveform_tensor.shape[1]], dtype=torch.int64),
        )
        feature_length = min(int(lengths[0]), self.config.max_frames)
        if feature_length <= 0:
            raise ValueError(f'No valid frontend frames generated: {audio_file}')
        output = np.zeros(
            (1, self.config.max_frames, self.config.n_mels * self.config.lfr_m), dtype=np.float32
        )
        output[0, :feature_length] = features[0, :feature_length].detach().cpu().numpy().astype(np.float32)
        return output, feature_length


@dataclass
class ParaformerConfig:
    """Configuration for the fixed-shape Paraformer S100 deployment.

    Attributes:
        encoder_model_path: Encoder HBM path.
        predictor_model_path: Predictor HBM path.
        decoder_model_path: Decoder HBM path.
        tokens_path: JSON token-list path.
        cmvn_path: FunASR CMVN file used to preprocess WAV input.
        max_label_len: Maximum decoder token count fixed during export.
    """

    encoder_model_path: str = '/opt/hobot/model/s100/basic/paraformer/paraformer_large_encoder_400x560_s100.hbm'
    predictor_model_path: str = '/opt/hobot/model/s100/basic/paraformer/paraformer_large_predictor_400x512_s100.hbm'
    decoder_model_path: str = '/opt/hobot/model/s100/basic/paraformer/paraformer_large_decoder_400x512_s100.hbm'
    tokens_path: str = '/opt/hobot/model/s100/basic/paraformer/tokens.json'
    cmvn_path: str = '/opt/hobot/model/s100/basic/paraformer/am.mvn'
    max_label_len: int = 100


class Paraformer:
    """Run the validated Paraformer multi-HBM inference pipeline on S100."""

    def __init__(self, config: ParaformerConfig) -> None:
        """Load three HBM models and resolve their model-specific tensor names.

        Args:
            config: Immutable paths and static decoder configuration.
        """
        self.config = config
        self.vocab = json.loads(Path(config.tokens_path).read_text(encoding='utf-8'))
        self.frontend = ParaformerFrontend(ParaformerFrontendConfig(cmvn_path=config.cmvn_path))
        self.runtime = HB_HBMRuntime([
            config.encoder_model_path,
            config.predictor_model_path,
            config.decoder_model_path,
        ])
        self._resolve_tensor_names()
        self.bias_embed = np.zeros((1, 1, 512), dtype=np.float32)

    def _resolve_tensor_names(self) -> None:
        """Resolve model and tensor names from HBM metadata without changing graph I/O."""
        names = self.runtime.model_names
        self.encoder_name = next(name for name in names if 'encoder' in name.lower())
        self.predictor_name = next(name for name in names if 'predictor' in name.lower())
        self.decoder_name = next(name for name in names if 'decoder' in name.lower())
        self.encoder_output = self.runtime.output_names[self.encoder_name][0]
        predictor_outputs = self.runtime.output_names[self.predictor_name]
        self.alphas_output = next(name for name in predictor_outputs if 'Add_output' in name)
        self.concat5_output = next(name for name in predictor_outputs if 'Concat_5' in name)
        decoder_inputs = self.runtime.input_names[self.decoder_name]
        self.decoder_encoder_input = next(name for name in decoder_inputs if 'after_norm' in name)
        self.decoder_token_num_input = next(name for name in decoder_inputs if name == 'token_num')
        self.decoder_bias_input = next(name for name in decoder_inputs if name == 'bias_embed')
        self.decoder_pre_acoustic_input = next(
            name for name in decoder_inputs if 'Shape_8609' in name or 'shape_8609' in name
        )
        self.decoder_logits_output = self.runtime.output_names[self.decoder_name][0]

    def set_scheduling_params(self, priority: int | None = None, bpu_cores: list[int] | None = None) -> None:
        """Accept standard scheduling parameters when supported by hbm_runtime.

        Args:
            priority: Optional runtime priority.
            bpu_cores: Optional BPU core affinity.
        """
        del priority, bpu_cores

    def pre_process(self, audio_file: str | Path) -> tuple[np.ndarray, int]:
        """Convert one 16 kHz WAV file into the fixed-shape encoder input.

        Args:
            audio_file: Path to a mono-compatible 16 kHz WAV file.

        Returns:
            Float32 tensor shaped ``[1, 400, 560]`` and its valid frame count.
        """
        return self.frontend.pre_process(audio_file)

    def forward(self, features: np.ndarray, feature_length: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Execute Encoder, Predictor, CPU CIF, and Decoder without reordering stages.

        Args:
            features: Fixed-shape encoder feature input.
            feature_length: Unpadded feature-frame count for CIF masking.

        Returns:
            Tuple of decoder logits, token counts, and per-stage milliseconds.
        """
        begin = perf_counter()
        result = self.runtime.run(features, model_name=self.encoder_name)
        encoder_output = result[self.encoder_name][self.encoder_output]
        encoder_ms = (perf_counter() - begin) * 1000

        begin = perf_counter()
        result = self.runtime.run(encoder_output, model_name=self.predictor_name)
        alphas = result[self.predictor_name][self.alphas_output]
        concat5 = result[self.predictor_name][self.concat5_output]
        predictor_ms = (perf_counter() - begin) * 1000

        begin = perf_counter()
        frame_fires, token_num = self._cif_numpy(alphas, concat5, feature_length)
        cif_ms = (perf_counter() - begin) * 1000

        begin = perf_counter()
        result = self.runtime.run({
            self.decoder_encoder_input: encoder_output,
            self.decoder_token_num_input: token_num,
            self.decoder_bias_input: self.bias_embed,
            self.decoder_pre_acoustic_input: frame_fires,
        }, model_name=self.decoder_name)
        decoder_ms = (perf_counter() - begin) * 1000
        logits = result[self.decoder_name][self.decoder_logits_output]
        return logits, token_num, {
            'encoder_ms': encoder_ms,
            'predictor_ms': predictor_ms,
            'cif_ms': cif_ms,
            'decoder_ms': decoder_ms,
        }

    def post_process(self, logits: np.ndarray, token_num: np.ndarray) -> str:
        """Apply greedy decoding, remove special tokens, and merge BPE subwords.

        Args:
            logits: Decoder logits shaped ``[1, max_label_len, vocab_size]``.
            token_num: Valid token count emitted by CIF.

        Returns:
            UTF-8 recognition text.
        """
        token_count = int(token_num[0])
        ids = np.argmax(logits[0, :token_count], axis=-1)
        tokens = [
            token for token_id in ids
            if not ((token := self.vocab[int(token_id)]).startswith('<') and token.endswith('>'))
        ]
        return ''.join(token.replace('@@', '') for token in tokens)

    def predict(self, audio_file: str | Path) -> tuple[str, dict[str, float]]:
        """Run one complete Paraformer inference request.

        Args:
            audio_file: Path to one mono-compatible 16 kHz WAV file.

        Returns:
            Recognition text and stage timing values in milliseconds.
        """
        features, feature_length = self.pre_process(audio_file)
        logits, token_num, timings = self.forward(features, feature_length)
        return self.post_process(logits, token_num), timings

    def __call__(self, audio_file: str | Path) -> tuple[str, dict[str, float]]:
        """Alias for :meth:`predict`.

        Args:
            audio_file: Path to one 16 kHz WAV file.

        Returns:
            Recognition text and timing data.
        """
        return self.predict(audio_file)

    def _cif_numpy(self, alphas: np.ndarray, concat5: np.ndarray, real_t: int) -> tuple[np.ndarray, np.ndarray]:
        """Run the original CPU NumPy CIF algorithm with padding masked after ``real_t``."""
        alphas = alphas.copy()
        if real_t < alphas.shape[1]:
            alphas[:, real_t:] = 0.0
        batch_size, _ = alphas.shape
        hidden_size = concat5.shape[-1]
        prefix_sum = np.cumsum(alphas.astype(np.float64), axis=1).astype(np.float32)
        prefix_sum_floor = np.floor(prefix_sum)
        displaced_floor = np.floor(np.roll(prefix_sum, 1, axis=1))
        displaced_floor[:, 0] = 0
        fire_indices = (prefix_sum_floor - displaced_floor) > 0
        fires = np.zeros_like(prefix_sum)
        fires[fire_indices] = 1.0
        fires += prefix_sum - prefix_sum_floor
        prefix_hidden = np.cumsum(alphas[..., None].astype(np.float64) * concat5.astype(np.float64), axis=1).astype(np.float32)
        frames = prefix_hidden[fire_indices]
        shifted_frames = np.roll(frames, 1, axis=0)
        batch_len = fire_indices.sum(axis=1)
        batch_indices = np.cumsum(batch_len)
        shifted_batch_indices = np.roll(batch_indices, 1)
        shifted_batch_indices[0] = 0
        shifted_frames[shifted_batch_indices] = 0
        remains = fires - np.floor(fires)
        remain_frames = remains[fire_indices][:, None] * concat5[fire_indices]
        shifted_remain_frames = np.roll(remain_frames, 1, axis=0)
        shifted_remain_frames[shifted_batch_indices] = 0
        frames = frames - shifted_frames + shifted_remain_frames - remain_frames
        frame_fires = np.zeros((batch_size, self.config.max_label_len, hidden_size), dtype=np.float32)
        slots = np.arange(self.config.max_label_len)[None, :]
        clamped_length = np.clip(batch_len, None, self.config.max_label_len)
        slot_mask = slots < clamped_length[:, None]
        frame_fires[slot_mask] = frames[:int(slot_mask.sum())]
        return frame_fires, clamped_length.astype(np.int32)
