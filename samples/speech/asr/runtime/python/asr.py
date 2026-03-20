# Copyright (c) 2025 D-Robotics Corporation
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

# flake8: noqa: E501
# flake8: noqa: E402

"""ASR (Automatic Speech Recognition) inference wrapper and pipeline utilities.

This module defines an ASR runtime wrapper built on HBM runtime.
It includes a configuration dataclass and a complete inference pipeline
(preprocess, forward, postprocess) for streaming speech-to-text transcription.

Key Features:
    - Streaming audio preprocessing: chunked reading, mono conversion,
      resampling, and z-score normalization
    - Greedy CTC decoding with vocabulary lookup
    - Supports RDK S100 and RDK S600 platforms

Typical Usage:
    >>> from asr import ASR, ASRConfig
    >>> config = ASRConfig(model_path='/opt/hobot/model/s100/basic/asr.hbm')
    >>> model = ASR(config)
    >>> text = model.predict('test_data/chi_sound.wav', id2token)

Notes:
    - Audio input is processed in fixed-length chunks of `audio_maxlen` samples
      at `new_rate` Hz, enabling streaming transcription of arbitrary-length audio.
"""

import os
import sys
import json
import hbm_runtime
import numpy as np
import soundfile as sf
from dataclasses import dataclass
from typing import Dict, Optional, Generator

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
import utils.py_utils.nn_math as nn_math


@dataclass
class ASRConfig:
    """Configuration for initializing the ASR model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing and inference in the ASR pipeline.

    Attributes:
        model_path: Path to the compiled ASR `.hbm` model.
        audio_maxlen: Number of audio samples per inference chunk (at new_rate Hz).
            Determines the fixed-length window fed to the model.
        new_rate: Target audio sample rate in Hz. Input audio is resampled to this
            rate before inference.

    """
    model_path: str = '/opt/hobot/model/s100/basic/asr.hbm'  # override at runtime with soc-specific path
    audio_maxlen: int = 30000
    new_rate: int = 16000


class ASR:
    """ASR inference wrapper based on HB_HBMRuntime.

    This class provides a complete inference pipeline for Automatic Speech
    Recognition, including streaming audio preprocessing, model execution,
    and CTC greedy decoding.

    Args:
        config: Configuration object containing model path and preprocessing
            parameters. All field semantics are defined in `ASRConfig`.

    Attributes:
        model: Loaded HBM runtime model handle.
        model_name: Name of the loaded model.
        input_names: List of input tensor names.
        output_name: Name of the primary output tensor.
        output_quants: Output quantization parameters.
        cfg: ASRConfig object with runtime parameters.

    """

    def __init__(self, config: ASRConfig) -> None:
        """Initialize the ASR model with the given configuration.

        Args:
            config: Configuration object. All field semantics are defined
                in `ASRConfig`.
        """
        # Load model and extract metadata
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)

        self.model_name = self.model.model_names[0]
        self.input_names = self.model.input_names[self.model_name]
        self.output_name = self.model.output_names[self.model_name][0]
        self.output_quants = self.model.output_quants[self.model_name]

        self.cfg = config

    def set_scheduling_params(self,
                              priority: Optional[int] = None,
                              bpu_cores: Optional[list] = None) -> None:
        """Configure inference scheduling parameters.

        Args:
            priority: Inference priority in the range [0, 255].
                Higher values mean higher priority.
            bpu_cores: List of BPU core indices used for inference.

        Returns:
            None
        """
        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}

        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    def pre_process(self,
                    audio_file: str,
                    audio_maxlen: Optional[int] = None,
                    new_rate: Optional[int] = None
                    ) -> Generator[Dict[str, Dict[str, np.ndarray]], None, None]:
        """Preprocess an audio file and yield input tensor dicts for each chunk.

        Reads the audio file in fixed-length chunks, applies mono conversion,
        resampling to `new_rate`, z-score normalization, and padding/truncation
        to `audio_maxlen`. Each chunk is yielded as a model-ready input tensor
        dict.

        Args:
            audio_file: Path to the input audio file (.wav or .flac).
            audio_maxlen: Number of samples per chunk. If None, uses the value
                from the configuration.
            new_rate: Target sample rate in Hz. If None, uses the value from
                the configuration.

        Yields:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`, ready to be passed to
            `forward()`. The tensor has shape `(1, audio_maxlen)`.

        Raises:
            ValueError: If the audio file cannot be read.
        """
        if audio_maxlen is None:
            audio_maxlen = self.cfg.audio_maxlen
        if new_rate is None:
            new_rate = self.cfg.new_rate

        with sf.SoundFile(audio_file, 'r') as f:
            orig_sr = f.samplerate

            # Frames to read per chunk at the original sample rate
            read_size = int(np.ceil(audio_maxlen * orig_sr / new_rate))

            while True:
                data = f.read(read_size, dtype='float32')
                if len(data) == 0:
                    break

                # Convert stereo/multi-channel to mono
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)

                # Resample to target rate if needed
                if orig_sr != new_rate:
                    from scipy.signal import resample
                    samples = round(len(data) * float(new_rate) / orig_sr)
                    data = resample(data, samples)

                # Z-score normalization
                data = nn_math.zscore_normalize_lastdim(data)

                # Pad or truncate to fixed length
                if len(data) < audio_maxlen:
                    pad = np.zeros(audio_maxlen - len(data), dtype=np.float32)
                    data = np.concatenate([data, pad])
                else:
                    data = data[:audio_maxlen]

                # Add batch dimension: (1, audio_maxlen)
                tensor = data[None, :].astype(np.float32)

                yield {
                    self.model_name: {
                        self.input_names[0]: tensor
                    }
                }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference on a single audio chunk.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`, in the form `{model_name: {input_name: tensor}}`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime,
            in the form `{model_name: {output_name: tensor}}`.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self,
                     outputs: Dict[str, Dict[str, np.ndarray]],
                     id2token: Dict[int, str]) -> str:
        """Decode model output logits into transcribed text for one chunk.

        Applies greedy (argmax) decoding over the time axis and maps token
        IDs to token strings via `id2token`. Removes `<pad>` tokens from
        the result.

        Args:
            outputs: Raw output tensors from `forward()`, in the form
                `{model_name: {output_name: tensor}}`.
            id2token: Vocabulary mapping from token ID (int) to token string.

        Returns:
            Transcribed text string for this chunk, with `<pad>` tokens removed.
        """
        logits = outputs[self.model_name][self.output_name]

        # Greedy decode: argmax over vocab dimension at each time step
        prediction = np.argmax(logits, axis=-1)

        # Map token IDs to characters and remove padding
        text = "".join([id2token[i] for i in prediction[0]])
        return text.replace("<pad>", "")

    def predict(self,
                audio_file: str,
                id2token: Dict[int, str],
                audio_maxlen: Optional[int] = None,
                new_rate: Optional[int] = None) -> str:
        """Run the complete ASR pipeline on an audio file.

        Iterates over all audio chunks via `pre_process`, runs `forward` and
        `post_process` on each chunk, and concatenates the results.

        Args:
            audio_file: Path to the input audio file (.wav or .flac).
            id2token: Vocabulary mapping from token ID to token string.
            audio_maxlen: Number of samples per chunk override.
            new_rate: Target sample rate override.

        Returns:
            Full transcribed text string for the entire audio file.
        """
        full_text = ""
        for input_tensor in self.pre_process(audio_file, audio_maxlen, new_rate):
            outputs = self.forward(input_tensor)
            full_text += self.post_process(outputs, id2token)
        return full_text

    def __call__(self,
                 audio_file: str,
                 id2token: Dict[int, str],
                 audio_maxlen: Optional[int] = None,
                 new_rate: Optional[int] = None) -> str:
        """Callable interface for the ASR pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            audio_file: Path to the input audio file.
            id2token: Vocabulary mapping from token ID to token string.
            audio_maxlen: Number of samples per chunk override.
            new_rate: Target sample rate override.

        Returns:
            Same return value as `predict()`.
        """
        return self.predict(audio_file, id2token, audio_maxlen, new_rate)
