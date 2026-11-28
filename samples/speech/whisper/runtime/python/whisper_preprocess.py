# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audio loading, log-Mel features and token decoding for Whisper Tiny."""

from __future__ import annotations

import base64
from pathlib import Path
import sys

import numpy as np

# Board entry points run directly from the sample without installing the zoo.
sys.path.append(str(Path(__file__).resolve().parents[5] / "utils/py_utils"))
import audio_utils

# Calibration callers also use the shared decoder for in-memory audio.
read_audio = audio_utils.read_audio

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_MEL_FRAMES = 3000

END_TOKEN = 50257
TASK_EN = 50259
TASK_ZH = 50260
SOT = 50258
TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
TIMESTAMP_BEGIN = 50364


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load and fit audio to the Whisper encoder's input length.

    Args:
        audio_path: Path to an audio recording.
        target_sr: Output sample rate in hertz. Whisper expects 16 kHz.

    Returns:
        A mono float32 waveform of N_SAMPLES samples.
    """
    return audio_utils.load_waveform(audio_path, target_sr, N_SAMPLES)


def load_mel_filters(filters_path: str) -> np.ndarray:
    """Load the precomputed Mel filterbank matrix from a text file.

    Args:
        filters_path: Path to the Mel filterbank text file.

    Returns:
        A float32 array of shape ``(80, 201)``.
    """
    data = np.loadtxt(filters_path, dtype=np.float32)
    return data.reshape(N_MELS, N_FFT // 2 + 1)


def log_mel_spectrogram(
    audio: np.ndarray,
    mel_filters: np.ndarray,
) -> np.ndarray:
    """Compute Whisper log-Mel features with a periodic Hann window.

    Args:
        audio: Mono 16 kHz waveform.
        mel_filters: Matrix of Mel weights with N_FFT // 2 + 1 columns.

    Returns:
        A float32 matrix indexed by Mel channel and time frame.
    """
    audio = np.asarray(audio, dtype=np.float32)
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)
    frames = windows[::HOP_LENGTH][:-1]
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    spectrum = np.fft.rfft(frames * window, axis=-1)
    power = np.abs(spectrum).astype(np.float32)**2
    mel = np.einsum("mf,tf->mt", mel_filters, power)
    log_spec = np.log10(np.maximum(mel, 1e-10))
    return (np.maximum(log_spec, log_spec.max() - 8.0) + 4.0) / 4.0


def pad_or_trim_mel(mel: np.ndarray,
                    max_frames: int = MAX_MEL_FRAMES) -> np.ndarray:
    """Pad or trim the mel spectrogram to a fixed number of time frames.

    Args:
        mel: Mel spectrogram of shape ``(n_mels, T)``.
        max_frames: Target number of time frames.

    Returns:
        A float32 array of shape ``(n_mels, max_frames)``.
    """
    result = np.zeros((N_MELS, max_frames), dtype=np.float32)
    actual = min(mel.shape[1], max_frames)
    result[:, :actual] = mel[:, :actual]
    return result


def encoder_input(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Build the fixed-size input used for both X5 calibration and inference.

    Args:
        audio: Mono 16 kHz waveform of any length.
        mel_filters: The encoder's (80, 201) Mel filterbank.

    Returns:
        A float32 tensor of shape (1, 80, 3000, 1).
    """
    waveform = audio_utils.fit_waveform(audio, N_SAMPLES)
    mel = log_mel_spectrogram(waveform, mel_filters)
    return mel[np.newaxis, :, :, np.newaxis]


def read_vocab(vocab_path: str) -> dict[str, str]:
    """Read a Whisper vocabulary file mapping token IDs to strings.

    The file format is one entry per line: ``<token_id> <token_str>``.
    Lines with only a token ID (no string) map to an empty string.

    Args:
        vocab_path: Path to the vocabulary text file.

    Returns:
        A dict mapping token ID strings to token strings.
    """
    vocab: dict[str, str] = {}
    with open(vocab_path, encoding="utf-8") as vocab_file:
        for line in vocab_file:
            parts = line.strip().split(" ")
            if len(parts) < 2:
                vocab[parts[0]] = ""
            else:
                vocab[parts[0]] = parts[1]
    return vocab


def base64_decode(encoded_string: str) -> str:
    """Decode text from a legacy base64 vocabulary.

    Args:
        encoded_string: Base64-encoded UTF-8 token text.

    Returns:
        Unicode text with invalid UTF-8 bytes replaced.
    """
    return base64.b64decode(encoded_string).decode("utf-8", errors="replace")
