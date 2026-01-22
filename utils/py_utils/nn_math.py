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

import cv2
import numpy as np

"""
nn_math: Lightweight neural network math utilities.

This module provides small, reusable numerical helpers commonly used in
neural network preprocessing or postprocessing, such as activation functions
and simple normalization operations. The implementations are framework-agnostic
and intended for clarity and portability.
"""


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid activation function.

    Applies the sigmoid function element-wise to the input NumPy array.

    Args:
        x: Input NumPy array.

    Returns:
        A NumPy array with the sigmoid function applied element-wise.
    """
    return 1.0 / (1.0 + cv2.exp(-x))


def zscore_normalize_lastdim(x: np.ndarray) -> np.ndarray:
    """Normalize the input array along the last dimension using z-score.

    This function performs standard score normalization (z-score) on the
    last dimension of the input array, resulting in zero mean and unit
    variance for each vector.

    Args:
        x: Input NumPy array with shape `(..., channels)`.

    Returns:
        A NumPy array where each vector along the last dimension is
        normalized to have mean 0 and variance 1.
    """
    mean = np.mean(x, axis=-1, keepdims=True)        # Compute mean per sample
    var = np.var(x, axis=-1, keepdims=True)          # Compute variance per sample
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))  # Z-score normalization
