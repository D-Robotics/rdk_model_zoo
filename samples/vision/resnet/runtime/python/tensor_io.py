"""Image preprocessing and NV12 tensor packing for the ResNet pilot.

This module is intentionally independent of the board runtime.  It translates
an OpenCV BGR image into the physical input arrays required by a validated
``ModelBinding`` and records the actual geometry used for the transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import cv2
import numpy as np

from samples.vision.resnet.runtime.python.model_binding import ModelBinding


@dataclass(frozen=True)
class ImageTransform:
    """The realized image geometry, including integer resize rounding and pads."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale_x: float
    scale_y: float
    crop_x: float = 0.0
    crop_y: float = 0.0


@dataclass(frozen=True)
class PreparedInput:
    """Runtime input tensors and the transform that produced them."""

    tensors: Mapping[str, np.ndarray]
    transform: ImageTransform


def resize_bgr(
    image: np.ndarray,
    input_width: int,
    input_height: int,
    *,
    resize_type: int = 1,
    interpolation: str | int = "nearest",
    letterbox_interpolation: str | int = "linear",
) -> tuple[np.ndarray, ImageTransform]:
    """Resize one BGR image using the source sample's two policies.

    ``resize_type=0`` stretches directly to the target geometry.  ``1`` keeps
    aspect ratio, uses the source helper's integer dimensions, and pads with
    BGR value 127.  The old helper leaves its letterbox cv2.resize call at
    OpenCV's default INTER_LINEAR even when the caller supplies a different
    direct-resize interpolation; that behavior is represented explicitly by
    the two interpolation arguments.
    """

    _validate_image(image)
    if input_height <= 0 or input_width <= 0:
        raise ValueError("Model input dimensions must be positive.")
    if input_height % 2 or input_width % 2:
        raise ValueError("NV12 model input dimensions must be even.")
    height, width = image.shape[:2]
    direct_interpolation = _cv_interpolation(interpolation)

    if resize_type == 0:
        resized = cv2.resize(
            image, (input_width, input_height), interpolation=direct_interpolation)
        transform = ImageTransform(
            original_height=height,
            original_width=width,
            resized_height=input_height,
            resized_width=input_width,
            pad_top=0,
            pad_bottom=0,
            pad_left=0,
            pad_right=0,
            scale_x=input_width / width,
            scale_y=input_height / height,
        )
        return np.ascontiguousarray(resized), transform

    if resize_type != 1:
        raise ValueError(f"Invalid resize_type: {resize_type}; expected 0 or 1.")

    scale = min(input_height / height, input_width / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=_cv_interpolation(letterbox_interpolation),
    )
    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(127, 127, 127),
    )
    transform = ImageTransform(
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        scale_x=resized_width / width,
        scale_y=resized_height / height,
    )
    return np.ascontiguousarray(padded), transform


def bgr_to_nv12_planes(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use the shared I420-to-NV12 conversion; preserve this entry's checks."""
    _validate_image(image)
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("NV12 conversion requires even image dimensions.")
    from samples._shared.image import bgr_to_nv12_planes as convert

    return convert(image)


def prepare_nv12(image: np.ndarray, binding: ModelBinding, *,
                 resize_type: Optional[int] = None) -> PreparedInput:
    """Prepare a BGR image for one validated packed or split NV12 binding."""

    contract = binding.contract
    chosen_resize = contract.resize_type if resize_type is None else resize_type
    resized, transform = resize_bgr(
        image,
        contract.input_width,
        contract.input_height,
        resize_type=chosen_resize,
        interpolation=contract.resize_interpolation,
        letterbox_interpolation=contract.letterbox_interpolation,
    )
    y, uv = bgr_to_nv12_planes(resized)
    if contract.input_protocol == "packed_nv12":
        if len(binding.input_names) != 1:
            raise ValueError("Packed NV12 binding must contain one input name.")
        packed = np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(
            1, contract.input_height * 3 // 2, contract.input_width, 1)
        tensors = {binding.input_names[0]: np.ascontiguousarray(packed, dtype=np.uint8)}
    elif contract.input_protocol == "split_nv12":
        if not binding.y_input_name or not binding.uv_input_name:
            raise ValueError("Split NV12 binding is missing Y/UV input roles.")
        tensors = {
            binding.y_input_name: y,
            binding.uv_input_name: uv,
        }
    else:
        raise ValueError(f"Unsupported input protocol {contract.input_protocol!r}.")

    prepared = PreparedInput(tensors=tensors, transform=transform)
    validate_input_tensors(binding, prepared.tensors)
    return prepared


def pack_nv12(image: np.ndarray, binding: ModelBinding, *,
              resize_type: Optional[int] = None) -> PreparedInput:
    """Compatibility name for :func:`prepare_nv12`."""

    return prepare_nv12(image, binding, resize_type=resize_type)


def pack_nv12_single(image: np.ndarray, input_width: int = 224,
                     input_height: int = 224, *, resize_type: int = 1,
                     interpolation: str | int = "nearest") -> np.ndarray:
    """Pack an image into the physical single-buffer NV12 shape used by X5."""

    resized, _ = resize_bgr(
        image,
        input_width,
        input_height,
        resize_type=resize_type,
        interpolation=interpolation,
        letterbox_interpolation="linear",
    )
    y, uv = bgr_to_nv12_planes(resized)
    return np.ascontiguousarray(np.concatenate((y.reshape(-1), uv.reshape(-1))).reshape(
        1, input_height * 3 // 2, input_width, 1), dtype=np.uint8)


def pack_nv12_planes(image: np.ndarray, input_width: int = 224,
                     input_height: int = 224, *, resize_type: int = 1,
                     interpolation: str | int = "nearest") -> tuple[np.ndarray, np.ndarray]:
    """Prepare the separate Y/UV arrays used by the S-series wrappers."""

    resized, _ = resize_bgr(
        image,
        input_width,
        input_height,
        resize_type=resize_type,
        interpolation=interpolation,
        letterbox_interpolation="linear",
    )
    return bgr_to_nv12_planes(resized)


def validate_input_tensors(binding: ModelBinding,
                           tensors: Mapping[str, np.ndarray]) -> None:
    """Reject missing, extra, or physically mis-shaped input arrays."""

    if set(tensors) != set(binding.input_names):
        raise ValueError(
            f"Input names do not match binding: expected {binding.input_names}, "
            f"got {tuple(tensors)}.")
    contract = binding.contract
    if contract.input_protocol == "packed_nv12":
        actual = tensors[binding.input_names[0]]
        expected = (1, contract.input_height * 3 // 2, contract.input_width, 1)
        if tuple(actual.shape) != expected:
            raise ValueError(f"Packed NV12 input shape {actual.shape} != {expected}.")
    else:
        y = tensors.get(binding.y_input_name or "")
        uv = tensors.get(binding.uv_input_name or "")
        expected_y = (1, contract.input_height, contract.input_width, 1)
        expected_uv = (1, contract.input_height // 2, contract.input_width // 2, 2)
        if y is None or tuple(y.shape) != expected_y:
            raise ValueError(f"Y input shape {None if y is None else y.shape} != {expected_y}.")
        if uv is None or tuple(uv.shape) != expected_uv:
            raise ValueError(f"UV input shape {None if uv is None else uv.shape} != {expected_uv}.")
    for name, value in tensors.items():
        if value.dtype != np.uint8:
            raise ValueError(f"NV12 input {name!r} must be uint8, got {value.dtype}.")
        if not value.flags.c_contiguous:
            raise ValueError(f"NV12 input {name!r} must be contiguous.")


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array in OpenCV BGR format.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must have shape (height, width, 3) in BGR order.")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Input image dimensions must be positive.")
    if image.dtype != np.uint8:
        raise ValueError(f"Input image must be uint8, got {image.dtype}.")


def _cv_interpolation(value: str | int) -> int:
    if isinstance(value, int):
        return value
    key = str(value).strip().lower()
    choices = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
    }
    if key not in choices:
        raise ValueError(f"Unsupported resize interpolation {value!r}.")
    return choices[key]


__all__ = [
    "ImageTransform",
    "PreparedInput",
    "bgr_to_nv12_planes",
    "pack_nv12",
    "pack_nv12_planes",
    "pack_nv12_single",
    "prepare_nv12",
    "resize_bgr",
    "validate_input_tensors",
]
