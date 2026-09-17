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

"""Target-local OCR image and tensor preparation helpers."""

from __future__ import annotations

import numpy as np

from samples.vision.paddle_ocr.runtime.python.model_binding import OCRPair


def prepare_detection(image: np.ndarray, pair: OCRPair) -> dict[str, np.ndarray]:
    """Prepare one BGR image for the selected detector's NV12 protocol.

    X5 uses a single packed ``[1,960,640,1]`` buffer.  S100 exposes separate
    Y and UV planes.  The resize policy is part of the pair contract: X5 uses
    linear interpolation and S100 uses area interpolation.
    """

    _validate_bgr_image(image)
    import cv2

    interpolation = cv2.INTER_LINEAR if pair.target == "x5" else cv2.INTER_AREA
    resized = cv2.resize(
        image,
        (pair.detector.runtime_input_shapes[pair.detector.input_names[0]][2]
         if pair.target != "x5" else 640,
         pair.detector.runtime_input_shapes[pair.detector.input_names[0]][1]
         if pair.target != "x5" else 640),
        interpolation=interpolation,
    )
    y, uv = _bgr_to_nv12_planes(resized)

    if pair.target == "x5":
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1)), axis=0).reshape(
            pair.detector.runtime_input_shapes["x"]
        )
        return {"x": packed.astype(np.uint8, copy=False)}

    return {
        "x_y": y.astype(np.uint8, copy=False),
        "x_uv": uv.astype(np.uint8, copy=False),
    }


def prepare_recognition(image: np.ndarray, pair: OCRPair) -> dict[str, np.ndarray]:
    """Prepare one BGR crop as the audited RGB float32 NCHW input."""

    _validate_bgr_image(image)
    import cv2

    shape = pair.recognizer.runtime_input_shapes["x"]
    height, width = shape[2], shape[3]
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # Match both legacy Python wrappers' operation order: division produces a
    # float64 temporary and conversion to float32 happens before channel swap.
    resized = (resized / 255.0).astype(np.float32)
    rgb = resized[:, :, [2, 1, 0]]
    return {"x": rgb[None].transpose(0, 3, 1, 2)}


def _validate_bgr_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("OCR image must be a NumPy array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"OCR image must have shape (H,W,3), got {image.shape}.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("OCR image must have non-zero height and width.")
    if image.dtype != np.dtype("uint8"):
        raise ValueError(f"OCR image must be uint8 BGR, got {image.dtype}.")


def _bgr_to_nv12_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the shared I420-to-NV12 conversion; preserve this entry's checks."""
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV12 preparation requires even image dimensions.")
    from samples._shared.image import bgr_to_nv12_planes as convert

    return convert(image)


__all__ = ["prepare_detection", "prepare_recognition"]
