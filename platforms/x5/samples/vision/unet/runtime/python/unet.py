#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run UNet semantic segmentation with the RDK X5 Python BPU runtime.

The module exposes a configuration class and a model class following the
repository sample contract. Images are accepted in OpenCV BGR order and are
converted to packed NV12 before BPU inference.

Typical Usage:
    >>> config = UNetConfig(model_path="unet_resnet18_voc_512x512_nv12.bin")
    >>> model = UNet(config)
    >>> mask = model.predict(image)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils.py_utils.preprocess as pre_utils  # noqa: E402

DEFAULT_MODEL_PATH = (
    SCRIPT_DIR.parent.parent / "model" / "unet_resnet18_voc_512x512_nv12.bin"
)


@dataclass
class UNetConfig:
    """Configure UNet X5 inference.

    Args:
        model_path: Path to the X5 ``bayes-e`` compiled ``.bin`` model.
        input_width: Model input width in pixels.
        input_height: Model input height in pixels.
        num_classes: Number of semantic classes in the output.
    """

    model_path: str = str(DEFAULT_MODEL_PATH)
    input_width: int = 512
    input_height: int = 512
    num_classes: int = 21


class UNet:
    """Encapsulate UNet preprocessing, BPU inference, and postprocessing.

    Args:
        config: Model path and semantic segmentation contract.

    Attributes:
        model: Loaded ``HB_HBMRuntime`` instance.
        model_name: Name of the single model in the binary.
        input_name: Name of the packed NV12 model input.
        output_name: Name of the semantic-logit model output.
    """

    def __init__(self, config: UNetConfig):
        """Load the model and validate its runtime metadata.

        Args:
            config: UNet runtime configuration.

        Raises:
            FileNotFoundError: If the configured model does not exist.
            RuntimeError: If ``hbm_runtime`` is unavailable on the device.
            ValueError: If model I/O does not match the UNet contract.
        """

        model_path = Path(config.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if config.input_width <= 0 or config.input_height <= 0:
            raise ValueError("input dimensions must be positive")
        if config.input_width % 2 or config.input_height % 2:
            raise ValueError("NV12 input dimensions must be even")
        if config.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        try:
            from hbm_runtime import HB_HBMRuntime
        except ImportError as exc:
            raise RuntimeError(
                "hbm_runtime is required; run this sample on RDK X5 OS >= 3.5.0"
            ) from exc

        self.config = config
        self.model = HB_HBMRuntime(str(model_path))
        if self.model.model_count != 1:
            raise ValueError("UNet runtime requires exactly one model")
        self.model_name = self.model.model_names[0]
        input_names = self.model.input_names[self.model_name]
        output_names = self.model.output_names[self.model_name]
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError("UNet runtime requires exactly one input and one output")
        self.input_name = input_names[0]
        self.output_name = output_names[0]

        input_dtype = self.model.input_dtypes[self.model_name][self.input_name]
        self.input_dtype = getattr(input_dtype, "name", str(input_dtype))
        if self.input_dtype != "NV12":
            raise ValueError(f"expected NV12 input, got {self.input_dtype}")
        output_dtype = self.model.output_dtypes[self.model_name][self.output_name]
        self.output_dtype = getattr(output_dtype, "name", str(output_dtype))
        self.output_quant = self.model.output_quants[self.model_name][self.output_name]
        self.input_shape = self.model.input_shapes[self.model_name][self.input_name]
        self.output_shape = self.model.output_shapes[self.model_name][self.output_name]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set optional model priority and BPU core affinity.

        Args:
            priority: Scheduler priority in the range supported by the runtime.
            bpu_cores: BPU core indices assigned to this model.

        Returns:
            None.

        Notes:
            Supplying neither parameter has no side effect.
        """

        parameters: Dict[str, Dict[str, Any]] = {}
        if priority is not None:
            parameters["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            parameters["bpu_cores"] = {self.model_name: bpu_cores}
        if parameters:
            self.model.set_scheduling_params(**parameters)

    def pre_process(self, image: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Resize a BGR image and convert it to packed NV12.

        Args:
            image: OpenCV BGR uint8 image with shape ``[H, W, 3]``.

        Returns:
            Nested runtime input dictionary in the form
            ``{model_name: {input_name: tensor}}``.

        Raises:
            ValueError: If the image type, shape, or dtype is unsupported.
        """

        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have BGR shape [H, W, 3]")
        if image.dtype != np.uint8:
            raise ValueError("image dtype must be uint8")

        resized = cv2.resize(
            image,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        width = self.config.input_width
        height = self.config.input_height
        y_plane, uv_plane = pre_utils.bgr_to_nv12_planes(resized)
        packed = np.concatenate(
            (y_plane.reshape(-1), uv_plane.reshape(-1))
        ).reshape(1, height * 3 // 2, width, 1)
        tensor = np.ascontiguousarray(packed, dtype=np.uint8)
        return {self.model_name: {self.input_name: tensor}}

    def forward(self, inputs: Dict[str, Dict[str, np.ndarray]]) -> Any:
        """Execute one BPU forward pass.

        Args:
            inputs: Nested dictionary returned by :meth:`pre_process`.

        Returns:
            Direct output of ``HB_HBMRuntime.run``.
        """

        return self.model.run(inputs)

    def post_process(self, outputs: Any) -> np.ndarray:
        """Convert raw semantic logits into a class-index mask.

        Args:
            outputs: Direct nested output returned by :meth:`forward`.

        Returns:
            UInt8 class-index mask with shape ``[512, 512]``.

        Raises:
            ValueError: If the output shape does not represent this UNet model.
        """

        try:
            raw_output = outputs[self.model_name][self.output_name]
        except (KeyError, TypeError) as exc:
            raise ValueError("runtime output does not match model metadata") from exc
        array = self._dequantize_output(np.asarray(raw_output))
        if array.ndim == 4:
            if array.shape[0] != 1:
                raise ValueError(f"expected output batch 1, got {array.shape}")
            array = array[0]
        if array.ndim == 3 and array.shape[0] == self.config.num_classes:
            return array.argmax(axis=0).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == self.config.num_classes:
            return array.argmax(axis=-1).astype(np.uint8)
        raise ValueError(f"unsupported UNet output shape: {array.shape}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run preprocessing, BPU inference, and postprocessing.

        Args:
            image: OpenCV BGR uint8 image.

        Returns:
            UInt8 semantic class-index mask.
        """

        return self.post_process(self.forward(self.pre_process(image)))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Run :meth:`predict` using function-call syntax.

        Args:
            image: OpenCV BGR uint8 image.

        Returns:
            UInt8 semantic class-index mask.
        """

        return self.predict(image)

    def _dequantize_output(self, array: np.ndarray) -> np.ndarray:
        """Apply runtime scale and zero-point metadata when required."""

        quant_type = getattr(
            getattr(self.output_quant, "quant_type", None), "name", "NONE"
        )
        if quant_type != "SCALE":
            return array
        scale = np.asarray(self.output_quant.scale, dtype=np.float32)
        zero_point = np.asarray(self.output_quant.zero_point, dtype=np.float32)
        if scale.size == 1:
            return (
                array.astype(np.float32) - float(zero_point.reshape(-1)[0])
            ) * float(scale.reshape(-1)[0])
        axis = int(self.output_quant.axis)
        if axis < 0:
            axis += array.ndim
        shape = [1] * array.ndim
        shape[axis] = scale.size
        return (array.astype(np.float32) - zero_point.reshape(shape)) * scale.reshape(
            shape
        )


def voc_palette(num_classes: int = 21) -> np.ndarray:
    """Build the deterministic Pascal VOC color palette.

    Args:
        num_classes: Number of palette entries to generate.

    Returns:
        RGB uint8 palette with shape ``[num_classes, 3]``.
    """

    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for class_id in range(num_classes):
        value = class_id
        bit = 0
        while value:
            palette[class_id, 0] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    return palette


def colorize_mask(mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
    """Convert a class-index mask into an OpenCV BGR visualization.

    Args:
        mask: Two-dimensional semantic class-index mask.
        num_classes: Number of valid class identifiers.

    Returns:
        BGR uint8 visualization with shape ``[H, W, 3]``.

    Raises:
        ValueError: If the mask shape or class range is invalid.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if mask.size and (int(mask.min()) < 0 or int(mask.max()) >= num_classes):
        raise ValueError("mask contains an invalid class identifier")
    rgb = voc_palette(num_classes)[mask.astype(np.int64)]
    return np.ascontiguousarray(rgb[..., ::-1])
