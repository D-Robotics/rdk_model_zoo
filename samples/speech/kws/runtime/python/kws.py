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

"""KWS (Keyword Spotting) inference wrapper and pipeline utilities.

This module defines a KWS runtime wrapper built on HBM runtime.
It includes a configuration dataclass and a complete inference pipeline
(preprocess, forward, postprocess) for detecting wake words in audio.

Key Features:
    - Audio preprocessing: truncation/padding to fixed length,
      fbank feature extraction via paddleaudio
    - Sigmoid-based confidence scoring for keyword detection
    - Supports RDK S100 and RDK S600 platforms

Typical Usage:
    >>> from kws import KWS, KWSConfig
    >>> config = KWSConfig(model_path='/opt/hobot/model/s100/basic/kws.hbm')
    >>> model = KWS(config)
    >>> score = model.predict('test_data/sample.wav')
"""

import os
import sys
import hbm_runtime
import numpy as np
import paddle
import paddleaudio
from paddleaudio.compliance.kaldi import fbank
from dataclasses import dataclass
from typing import Dict, Optional

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io

# Maximum audio length in samples (60 seconds at 16 kHz)
THRES = 60000


@dataclass
class KWSConfig:
    """Configuration for initializing the KWS model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing and inference in the KWS pipeline.

    Attributes:
        model_path: Path to the compiled KWS `.hbm` model.
        audio_maxlen: Maximum number of audio samples before truncation.
        frame_shift: Frame shift in milliseconds for fbank extraction.
        frame_length: Frame length in milliseconds for fbank extraction.
        n_mels: Number of mel filter banks for fbank extraction.
    """
    model_path: str = '/opt/hobot/model/s100/basic/kws.hbm'
    audio_maxlen: int = THRES
    frame_shift: int = 10
    frame_length: int = 25
    n_mels: int = 80


class KWS:
    """KWS inference wrapper based on HB_HBMRuntime.

    This class provides a complete inference pipeline for Keyword Spotting,
    including audio preprocessing with fbank feature extraction, model
    execution, and confidence score computation.

    Args:
        config: Configuration object containing model path and preprocessing
            parameters. All field semantics are defined in `KWSConfig`.

    Attributes:
        model: Loaded HBM runtime model handle.
        model_name: Name of the loaded model.
        input_names: List of input tensor names.
        output_name: Name of the primary output tensor.
        output_quants: Output quantization parameters.
        cfg: KWSConfig object with runtime parameters.
    """

    def __init__(self, config: KWSConfig) -> None:
        """Initialize the KWS model with the given configuration.

        Args:
            config: Configuration object. All field semantics are defined
                in `KWSConfig`.
        """
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

    @staticmethod
    def audio_trunc(audio_arr: paddle.Tensor, thres: int = THRES) -> paddle.Tensor:
        """Truncate or zero-pad audio to a fixed length.

        Args:
            audio_arr: 2D paddle tensor of shape (1, N) containing audio samples.
            thres: Target length in samples.

        Returns:
            Paddle tensor of shape (1, thres), truncated or zero-padded.
        """
        length = audio_arr.shape[1]
        if length > thres:
            return audio_arr[:, :thres]
        elif length < thres:
            pad_zero = paddle.zeros((1, thres), dtype=audio_arr.dtype)
            pad_zero[:, :length] = audio_arr
            return pad_zero
        return audio_arr

    def pre_process(self, audio_file: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess an audio file into model-ready fbank features.

        Reads the audio file, truncates/pads to fixed length, and extracts
        fbank features using paddleaudio.

        Args:
            audio_file: Path to the input audio file (.wav).

        Returns:
            A nested input tensor dictionary in the form:
            `{model_name: {input_name: tensor}}`, ready to be passed to
            `forward()`. The tensor has shape `(1, 1, T, 80)` where T
            is the number of fbank frames.
        """
        waveform, sr = paddleaudio.load(audio_file)
        waveform = self.audio_trunc(waveform, self.cfg.audio_maxlen)

        feat = fbank(
            waveform=paddle.to_tensor(waveform),
            sr=sr,
            frame_shift=self.cfg.frame_shift,
            frame_length=self.cfg.frame_length,
            n_mels=self.cfg.n_mels,
        )
        tensor = feat.unsqueeze(0).numpy()

        return {
            self.model_name: {
                self.input_names[0]: tensor
            }
        }

    def forward(self, input_tensor: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute model inference on fbank features.

        Args:
            input_tensor: Preprocessed input tensor dictionary produced by
                `pre_process()`, in the form `{model_name: {input_name: tensor}}`.

        Returns:
            A dictionary containing raw output tensors returned by the runtime,
            in the form `{model_name: {output_name: tensor}}`.
        """
        outputs = self.model.run(input_tensor)
        return outputs

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> float:
        """Extract keyword confidence score from model output.

        Takes the sigmoid output and returns the maximum value as the
        keyword detection confidence score.

        Args:
            outputs: Raw output tensors from `forward()`, in the form
                `{model_name: {output_name: tensor}}`.

        Returns:
            Confidence score as a float in [0, 1]. Higher values indicate
            stronger keyword detection.
        """
        out = outputs[self.model_name][self.output_name]
        return float(np.max(out))

    def predict(self, audio_file: str) -> float:
        """Run the complete KWS pipeline on an audio file.

        Args:
            audio_file: Path to the input audio file (.wav).

        Returns:
            Keyword detection confidence score as a float.
        """
        input_tensor = self.pre_process(audio_file)
        outputs = self.forward(input_tensor)
        return self.post_process(outputs)

    def __call__(self, audio_file: str) -> float:
        """Callable interface for the KWS pipeline.

        This method is functionally equivalent to calling `predict()`.

        Args:
            audio_file: Path to the input audio file.

        Returns:
            Same return value as `predict()`.
        """
        return self.predict(audio_file)
