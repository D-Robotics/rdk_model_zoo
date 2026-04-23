"""UNet-resnet50 semantic segmentation model implementation.

This module provides a complete inference pipeline for UNet-resnet50
semantic segmentation on Horizon BPU platforms.

Key Features:
    - Model loading and metadata extraction
    - NV12 preprocessing for BPU input
    - Optimized argmax post-processing
    - Visualization with color mapping

Typical Usage:
    >>> from unet import UNetConfig, UNet
    >>> cfg = UNetConfig(model_path="/path/to/model.bin")
    >>> model = UNet(cfg)
    >>> mask = model.predict(image)

Notes:
    - Input image should be in BGR format (OpenCV default).
    - Output mask values are class indices (0 to num_classes-1).
    - The model expects NV12 input with shape [1, 3, 512, 512].
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn


class UNetConfig:
    """Configuration for UNet-resnet50 model initialization and inference.

    This class holds all hyperparameters and paths required to load and run
    the UNet-resnet50 semantic segmentation model.

    Attributes:
        model_path (str): Path to the BPU quantized *.bin model file.
        num_classes (int): Number of segmentation classes.
        input_size (Tuple[int, int]): Model input resolution (width, height).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 21,
        input_size: Tuple[int, int] = (512, 512),
    ):
        """Initialize configuration with default or user-provided values.

        Args:
            model_path: Path to the BPU model. If None, a local default path
                relative to the runtime directory is used.
            num_classes: Number of segmentation classes (including background).
            input_size: Model input resolution as (width, height).
        """
        if model_path is None:
            self.model_path = "../../model/unet_resnet50_512x512_nv12.bin"
        else:
            self.model_path = model_path

        self.num_classes = num_classes
        self.input_size = input_size


class UNet:
    """UNet-resnet50 semantic segmentation model.

    This class encapsulates the complete inference pipeline for UNet-resnet50,
    including preprocessing, forward inference, and post-processing.

    Args:
        config (UNetConfig): Configuration object containing model path and
            inference parameters.

    Attributes:
        config (UNetConfig): The configuration object.
        model: Loaded BPU model instance.
        input_name (str): Name of the model input tensor.
        output_name (str): Name of the model output tensor.
        input_w (int): Model input width.
        input_h (int): Model input height.
        out_c (int): Number of output channels (classes).
        out_h (int): Output feature map height.
        out_w (int): Output feature map width.
    """

    def __init__(self, config: UNetConfig):
        """Initialize the UNet model by loading the BPU model and extracting metadata."""
        self.config = config
        self.model = None
        self.input_name = ""
        self.output_name = ""
        self.input_w = 0
        self.input_h = 0
        self.out_c = 0
        self.out_h = 0
        self.out_w = 0

        self._load_model()

    def _load_model(self) -> None:
        """Load the BPU model and extract tensor metadata."""
        try:
            self.model = dnn.load(self.config.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {self.config.model_path}: {e}"
            ) from e

        # Extract input metadata
        inp = self.model[0].inputs[0]
        self.input_name = inp.name
        self.input_h = inp.properties.shape[2]
        self.input_w = inp.properties.shape[3]

        # Extract output metadata
        out = self.model[0].outputs[0]
        self.output_name = out.name
        self.out_c = out.properties.shape[1]
        self.out_h = out.properties.shape[2]
        self.out_w = out.properties.shape[3]

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        core_id: Optional[int] = None,
    ) -> None:
        """Set scheduling parameters for BPU inference.

        Args:
            priority: Inference priority. If None, no change is made.
            core_id: BPU core ID to bind. If None, no change is made.

        Notes:
            If all parameters are None, this function has no side effects.
        """
        # Scheduling parameters are not directly exposed in pyeasy_dnn.
        # Reserved for future extension or wrapper-specific implementations.
        pass

    def pre_process(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Preprocess an input image for BPU inference.

        Steps:
            1. Resize the image to the model input size.
            2. Convert BGR to NV12 format.

        Args:
            image: Input image in BGR format, shape [H, W, 3].

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping input tensor name to
                the preprocessed NV12 data. The NV12 array is flattened to 1D.

        Raises:
            ValueError: If the input is not a valid 3-channel image.
        """
        if image is None or len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(
                "Input must be a 3-channel BGR image with shape [H, W, 3]."
            )

        self.orig_h, self.orig_w = image.shape[:2]

        # Resize to model input size
        resized = cv2.resize(
            image,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR to NV12
        nv12 = self._bgr_to_nv12(resized)

        return {self.input_name: nv12}

    def _bgr_to_nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to NV12 format.

        Args:
            bgr_img: Input image in BGR format.

        Returns:
            np.ndarray: Flattened NV12 data.
        """
        height, width = bgr_img.shape[:2]
        area = height * width

        # BGR -> YUV_I420 (planar)
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape(
            (area * 3 // 2,)
        )

        # Split Y and UV
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))

        # Pack UV interleaved (NV12 format: YYYYYYYY UVUVUVUV)
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

        # Assemble NV12
        nv12 = np.zeros_like(yuv420p)
        nv12[:area] = y
        nv12[area:] = uv_packed

        return nv12

    def forward(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute a single forward inference pass.

        Args:
            inputs: Dictionary mapping input tensor name to preprocessed data.
                Must match the format returned by `pre_process`.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping output tensor name to
                the raw output tensor data.
        """
        input_tensor = inputs[self.input_name]
        outputs = self.model[0].forward(input_tensor)

        # Convert to dictionary format
        return {self.output_name: outputs[0].buffer}

    def post_process(
        self,
        outputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Convert raw model outputs to a semantic segmentation mask.

        Args:
            outputs: Dictionary mapping output tensor name to raw tensor data.
                Must match the format returned by `forward`.

        Returns:
            np.ndarray: Segmentation mask of shape [H, W] with dtype uint8,
                where each pixel value is a class index. The mask is resized
                back to the original input image dimensions.
        """
        output = outputs[self.output_name]

        # Reshape to [C, H, W]
        output = output.reshape(self.out_c, self.out_h, self.out_w)

        # Argmax to get per-pixel class index
        mask = np.argmax(output, axis=0).astype(np.uint8)

        # Resize back to original image size
        mask = cv2.resize(
            mask,
            (self.orig_w, self.orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        return mask

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run the complete inference pipeline on a single image.

        This is a convenience method that chains pre_process, forward, and
        post_process in sequence.

        Args:
            image: Input image in BGR format, shape [H, W, 3].

        Returns:
            np.ndarray: Segmentation mask of shape [H, W] with dtype uint8.
        """
        inputs = self.pre_process(image)
        outputs = self.forward(inputs)
        return self.post_process(outputs)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Enable callable model instance, equivalent to `predict`."""
        return self.predict(image)

    def visualize(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Visualize the segmentation result by overlaying a color mask.

        Args:
            image: Original input image in BGR format.
            mask: Segmentation mask with shape [H, W] and dtype uint8.
            alpha: Opacity of the overlay mask (0.0 to 1.0).

        Returns:
            np.ndarray: Blended visualization image in BGR format.
        """
        # Generate deterministic color palette
        np.random.seed(42)
        colors = np.random.randint(50, 255, (self.config.num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background is black

        # Create colored mask
        color_mask = np.zeros_like(image)
        for cls_id in range(self.config.num_classes):
            color_mask[mask == cls_id] = colors[cls_id]

        # Blend with original image
        result = cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0)

        # Add legend for first 5 classes
        for i in range(min(5, self.config.num_classes)):
            cv2.rectangle(result, (10, 30 + i * 25), (30, 50 + i * 25), colors[i].tolist(), -1)
            cv2.putText(
                result,
                f"Class {i}",
                (35, 45 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return result
