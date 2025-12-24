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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    @brief Compute the sigmoid activation function.
    @param x Input NumPy array.
    @return NumPy array after applying sigmoid function element-wise.
    """
    return 1.0 / (1.0 + cv2.exp(-x))


def zscore_normalize_lastdim(x: np.ndarray) -> np.ndarray:
    """
    @brief Normalize input array along the last dimension.
    @details This function performs standard score normalization (z-score).
    @param x Input NumPy array of shape (..., channels).
    @return Normalized array with mean 0 and variance 1 per vector.
    """
    mean = np.mean(x, axis=-1, keepdims=True)        # Compute mean per sample
    var = np.var(x, axis=-1, keepdims=True)          # Compute variance per sample
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))  # Z-score normalization
