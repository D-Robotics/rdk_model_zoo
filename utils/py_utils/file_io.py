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


"""
file_io: File and resource I/O utilities.

This module defines a collection of helper functions for interacting with
files and external resources. It serves as a common utility layer for
handling file existence checks, resource loading, and basic data access,
and is intended to be reused across different samples and runtimes.

Key Features:
    - Provide unified interfaces for file and resource access.
    - Encapsulate common file-related operations to reduce duplication.
    - Remain independent of specific models or tasks.

Typical Usage:
    >>> from file_io import download_model_if_needed
    >>> download_model_if_needed(model_path, download_url)

Notes:
    - This module focuses on general-purpose I/O utilities and should not
      include algorithm-specific or business-specific logic.
    - New file or resource related helpers should follow the same design
      principles and be added to this module when appropriate.
"""


import os
import cv2
import numpy as np
from typing import Dict

def download_model_if_needed(model_path: str, download_url: str) -> None:
    """Ensure that the model file exists locally, downloading it if necessary.

    If the model file specified by `model_path` does not exist, this function
    downloads it from the given URL and saves it to the target location.

    Args:
        model_path: Full local path where the model file should be saved.
        download_url: URL used to download the model file.

    Returns:
        None

    Raises:
        RuntimeError: If the download fails or the model file does not exist
            after the download attempt.
    """
    # Return immediately if the model file already exists
    if os.path.exists(model_path):
        print(f"Model already exists: {model_path}")
        return

    # Extract directory and filename from the target model path
    dirname = os.path.dirname(model_path)
    filename = os.path.basename(model_path)

    print(f"File {model_path} does not exist. Downloading model '{filename}'...")

    # Create the directory if it does not already exist
    os.makedirs(dirname, exist_ok=True)

    print(f"Downloading from: {download_url}")
    print(f"Saving to: {model_path}")

    # Execute the wget command to download the model file
    cmd = f"wget -c {download_url} -O {model_path}"
    ret = os.system(cmd)

    # Validate whether the download was successful
    if ret != 0 or not os.path.exists(model_path):
        raise RuntimeError(f"Failed to download model from {download_url}")


def load_image(img_path: str) -> np.ndarray:
    """Load an image from a file path using OpenCV.

    The image is loaded in BGR color format, which is the default format
    used by OpenCV.

    Args:
        img_path: Path to the image file.

    Returns:
        The loaded image as a NumPy array in BGR format.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the given path.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img


def load_class_names(path: str) -> list:
    """Load class names from a label file.

    Each line in the file is treated as a single class name. Empty lines
    are ignored.

    Args:
        path: Path to the label file. Each non-empty line contains one
            class name.

    Returns:
        A list of class name strings.
    """
    with open(path, 'r') as f:
        # Strip whitespace and filter out empty lines
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_labels(label_path: str) -> Dict[int, str]:
    """
    @brief Load labels from file. Supports dictionary format string or line-separated list.
    @param label_path Path to the label file.
    @return Dictionary of labels {index: name}.
    """
    labels = {}
    if not os.path.exists(label_path):
        print(f"Warning: Label file not found: {label_path}")
        return labels

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if content.startswith('{'):
            # Handle python dictionary string format (e.g. ImageNet labels)
            labels = eval(content)
        else:
            # Handle standard line-separated format
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            labels = {i: name for i, name in enumerate(lines)}
            
    except Exception as e:
        print(f"Warning: Failed to load labels from {label_path}: {e}")

    return labels