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

"""
CLIP image-text matching inference module.

This module implements the CLIP sample runtime used on RDK X5. The image
encoder is a compiled `.bin` model executed through `hbm_runtime`, while the
text encoder remains an ONNX model executed through `onnxruntime`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import hbm_runtime
import numpy as np
import onnxruntime as ort

from simple_tokenizer import SimpleTokenizer


@dataclass
class CLIPConfig:
    """
    Configuration for CLIP image-text matching.

    Args:
        image_model_path: Path to the BPU `.bin` image encoder.
        text_model_path: Path to the ONNX text encoder.
        context_length: Token context length used by the CLIP text encoder.
        image_size: Image encoder input resolution.
    """

    image_model_path: str
    text_model_path: str
    context_length: int = 77
    image_size: int = 224


class CLIPMatcher:
    """
    CLIP image-text matcher with a BPU image encoder and ONNX text encoder.

    The wrapper exposes the standard sample stages: `set_scheduling_params`,
    `pre_process`, `forward`, `post_process`, `predict`, and `__call__`.
    Text labels are supplied by the entry script and are not embedded in this
    module.
    """

    def __init__(self, config: CLIPConfig):
        """
        Initialize both CLIP encoder runtimes.

        Args:
            config: Runtime configuration containing image and text model paths.
        """

        self.cfg = config
        self.image_model = hbm_runtime.HB_HBMRuntime(config.image_model_path)
        self.image_model_name = self.image_model.model_names[0]
        self.input_names = self.image_model.input_names[self.image_model_name]
        self.output_names = self.image_model.output_names[self.image_model_name]
        self.text_session = ort.InferenceSession(config.text_model_path)
        self.text_input_name = self.text_session.get_inputs()[0].name
        self.text_output_name = self.text_session.get_outputs()[0].name
        self.tokenizer = SimpleTokenizer()

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """
        Set optional BPU scheduling parameters for the image encoder.

        Args:
            priority: Runtime priority in the range `0~255`.
            bpu_cores: BPU core indexes used by the image encoder.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.image_model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.image_model_name: bpu_cores}
        if kwargs:
            self.image_model.set_scheduling_params(**kwargs)

    def tokenize(self, texts: List[str], truncate: bool = False) -> np.ndarray:
        """
        Tokenize text prompts for the CLIP text encoder.

        Args:
            texts: Text prompt list.
            truncate: Whether to truncate prompts longer than context length.

        Returns:
            Token array with shape `(num_text, context_length)` and dtype int32.
        """

        sot_token = self.tokenizer.encoder["<|startoftext|>"]
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), self.cfg.context_length), dtype=np.int32)
        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.cfg.context_length:
                if not truncate:
                    raise RuntimeError(f"Input text is too long for context length {self.cfg.context_length}: {texts[i]}")
                tokens = tokens[: self.cfg.context_length]
                tokens[-1] = eot_token
            result[i, : len(tokens)] = np.asarray(tokens, dtype=np.int32)
        return result

    def pre_process(self, image: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Convert one BGR image into the image encoder tensor.

        Args:
            image: Input image in OpenCV BGR format.

        Returns:
            Nested input dictionary accepted by `hbm_runtime.run()`.
        """

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        if h < w:
            new_h = self.cfg.image_size
            new_w = int(round(w * self.cfg.image_size / h))
        else:
            new_w = self.cfg.image_size
            new_h = int(round(h * self.cfg.image_size / w))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        y0 = max((new_h - self.cfg.image_size) // 2, 0)
        x0 = max((new_w - self.cfg.image_size) // 2, 0)
        cropped = resized[y0 : y0 + self.cfg.image_size, x0 : x0 + self.cfg.image_size]
        tensor = cropped.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return {self.image_model_name: {self.input_names[0]: tensor}}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Execute the image encoder on BPU.

        Args:
            inputs: Prepared image encoder inputs.

        Returns:
            Image feature vector.
        """

        outputs = self.image_model.run(inputs)[self.image_model_name]
        return outputs[self.output_names[0]].reshape(-1).astype(np.float32)

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Execute the ONNX text encoder for a list of prompts.

        Args:
            texts: Prompt list to encode.

        Returns:
            Text feature matrix with one row per prompt.
        """

        tokens = self.tokenize(texts)
        return self.text_session.run([self.text_output_name], {self.text_input_name: tokens})[0].astype(np.float32)

    def post_process(self, image_feature: np.ndarray, text_features: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between image and text features.

        Args:
            image_feature: Image feature vector from the BPU image encoder.
            text_features: Text feature matrix from the ONNX text encoder.

        Returns:
            Similarity scores with shape `(num_text,)`.
        """

        image_norm = np.linalg.norm(image_feature) + 1e-12
        text_norm = np.linalg.norm(text_features, axis=1) + 1e-12
        return (text_features @ image_feature) / (text_norm * image_norm)

    def predict(self, image: np.ndarray, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the complete CLIP image-text matching pipeline.

        Args:
            image: Input image in OpenCV BGR format.
            texts: Candidate text prompts.

        Returns:
            A tuple containing similarity scores and descending rank indexes.
        """

        image_feature = self.forward(self.pre_process(image))
        text_features = self.encode_text(texts)
        scores = self.post_process(image_feature, text_features)
        order = np.argsort(scores)[::-1]
        return scores, order

    def __call__(self, image: np.ndarray, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provide functional-style access to `predict()`.

        Args:
            image: Input image in OpenCV BGR format.
            texts: Candidate text prompts.

        Returns:
            The same tuple returned by `predict()`.
        """

        return self.predict(image, texts)
