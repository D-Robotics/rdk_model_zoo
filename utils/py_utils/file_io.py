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

import os
import cv2
import numpy as np

def download_model_if_needed(model_path: str, download_url: str) -> None:
    """
    @brief Ensure the model file exists locally; download it automatically if missing.

    @param model_path Full local path where the model should be saved.
    @param download_url The URL used to download the model file.
    @return None
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
    """
    @brief Load an image from file path using OpenCV.
    @param img_path Path to the image file.
    @return Image as a NumPy ndarray in BGR format.
    @throws FileNotFoundError if the image cannot be loaded.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img


def load_class_names(path: str) -> list:
    """
    @brief Load class names from a file.
    @param path Path to the label file, each line contains a class name.
    @return List of class name strings.
    """
    with open(path, 'r') as f:
        # Strip whitespace and filter out empty lines
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names
