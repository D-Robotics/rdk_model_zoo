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
preprocess: Preprocessing utilities for vision model inputs.

This module provides reusable preprocessing helpers to convert raw inputs
into model-ready representations, including image resizing strategies and
format conversions. It is designed to be shared across multiple samples and
runtimes to keep preprocessing consistent and easy to maintain.

Key Features:
    - Convert between common image formats and planar representations.
    - Resize inputs with optional aspect-ratio preservation (e.g., letterbox).
    - Split and manipulate NV12 data into components suitable for inference.

Notes:
    - The module focuses on generic preprocessing building blocks; task- or
      model-specific preprocessing policies should be implemented at the
      sample level.
    - Helper coverage may evolve as new input formats and models are added.
"""


import cv2
import numpy as np


def bgr_to_nv12_planes(image: np.ndarray) -> tuple:
    """Convert a BGR image to NV12 format (Y and UV planes).

    This function converts a BGR image into NV12 format by first transforming
    it into planar YUV420 (I420) format and then interleaving the U and V
    planes to form the UV plane.

    Args:
        image: Input BGR image as a NumPy array with shape `(H, W, 3)`.

    Returns:
        A tuple containing:
            - y: Y plane with shape `(1, H, W, 1)`.
            - uv: UV plane with shape `(1, H/2, W/2, 2)`.
    """
    height, width = image.shape[:2]
    area = height * width

    # Convert to planar YUV I420 format
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    # Extract Y, U, V planes
    y = yuv420p[:area].reshape((height, width))
    u = yuv420p[area:area + area // 4].reshape((height // 2, width // 2))
    v = yuv420p[area + area // 4:].reshape((height // 2, width // 2))

    # Interleave U and V to form UV plane
    uv = np.stack((u, v), axis=-1)

    # Add batch and channel dimensions
    y = y[np.newaxis, :, :, np.newaxis]
    uv = uv[np.newaxis, :, :, :]

    return y, uv


def resized_image(img: np.ndarray, input_W: int, input_H: int,
                  resize_type: int = 1,
                  interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """Resize an image using direct resize or letterbox strategy.

    This function resizes the input image to the target resolution required
    by the model. It supports either direct resizing or letterbox resizing
    with padding to preserve the aspect ratio.

    Args:
        img: Input image array with shape `(H, W, 3)`.
        input_W: Target image width.
        input_H: Target image height.
        resize_type: Resize strategy used during preprocessing.
            - 0: Direct resize.
            - 1: Letterbox resize with padding to preserve aspect ratio.
        interpolation: OpenCV interpolation method used for resizing.

    Returns:
        The resized image with shape `(input_H, input_W, 3)`.

    Raises:
        ValueError: If an invalid `resize_type` is provided.
    """
    img_h, img_w = img.shape[:2]

    if resize_type == 0:  # Direct resize
        resized = cv2.resize(img, (input_W, input_H), interpolation=interpolation)
    elif resize_type == 1:  # Letterbox resize (preserve aspect ratio)
        scale = min(input_H / img_h, input_W / img_w)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        pad_w = input_W - new_w
        pad_h = input_H - new_h
        left, right = pad_w // 2, pad_w - pad_w // 2
        top, bottom = pad_h // 2, pad_h - pad_h // 2

        # Pad image with gray (127,127,127)
        resized = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=(127, 127, 127))
    else:
        raise ValueError(f"Invalid resize_type: {resize_type}, must be 0 or 1")

    return resized


def split_nv12_bytes(nv12_bytes: bytes, width: int, height: int) -> tuple:
    """Split raw NV12 bytes into Y and UV planes.

    This function parses a raw NV12-encoded byte stream and separates it
    into the Y (luma) plane and the interleaved UV (chroma) plane.

    Args:
        nv12_bytes: Raw NV12-encoded byte stream.
        width: Image width.
        height: Image height.

    Returns:
        A tuple containing:
            - y: Y plane with shape `(H, W)` and dtype `uint8`.
            - uv: Interleaved UV plane with shape `(H/2, W)` and dtype `uint8`.
    """
    y_size = width * height
    uv_size = y_size // 2
    nv12_array = np.frombuffer(nv12_bytes, dtype=np.uint8)

    y = nv12_array[:y_size].reshape((height, width))
    uv = nv12_array[y_size:y_size + uv_size].reshape((height // 2, width))

    return y, uv


def letterbox_resize_gray(gray_img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a grayscale image using letterbox (aspect-ratio preserving) strategy.

    This function resizes a grayscale image while preserving its aspect ratio,
    then pads the resized image with a constant gray value to match the target
    resolution.

    Args:
        gray_img: Input grayscale image with shape `(H, W)`.
        target_w: Target image width.
        target_h: Target image height.

    Returns:
        The resized and padded grayscale image with shape
        `(target_h, target_w)`.
    """
    h, w = gray_img.shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray_img, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    # Pad with value 127 (gray)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=127)
    return padded


def resize_nv12_yuv(y: np.ndarray, uv: np.ndarray,
                    target_h: int = 672, target_w: int = 672,
                    keep_ratio: bool = True) -> tuple:
    """Resize the Y and UV planes of an NV12 image to a target resolution.

    This function resizes the luma (Y) and chroma (UV) planes of an NV12 image.
    When `keep_ratio` is enabled, letterbox resizing is applied to preserve
    the original aspect ratio; otherwise, direct resizing is used.

    Args:
        y: Y (luma) plane with shape `(H, W)`.
        uv: Interleaved UV (chroma) plane with shape `(H/2, W)`.
        target_h: Target image height.
        target_w: Target image width.
        keep_ratio: Whether to preserve the aspect ratio using letterbox
            resizing. If `False`, direct resizing is applied.

    Returns:
        A tuple containing:
            - y_resized: Resized Y plane with shape `(target_h, target_w)`.
            - uv_resized: Resized UV plane with shape
              `(target_h/2, target_w/2, 2)`.
    """
    # Resize Y
    if keep_ratio:
        y_resized = letterbox_resize_gray(y, target_w, target_h)
    else:
        y_resized = cv2.resize(y, (target_w, target_h))

    # Split UV into U and V components
    u = uv[:, 0::2]
    v = uv[:, 1::2]

    # Resize U and V separately
    if keep_ratio:
        u_resized = letterbox_resize_gray(u, target_w // 2, target_h // 2)
        v_resized = letterbox_resize_gray(v, target_w // 2, target_h // 2)
    else:
        u_resized = cv2.resize(u, (target_w // 2, target_h // 2))
        v_resized = cv2.resize(v, (target_w // 2, target_h // 2))

    # Re-stack into UV plane
    uv_resized = np.stack((u_resized, v_resized), axis=-1)

    return y_resized, uv_resized
