"""EfficientSAM-Tiny dual-bin runtime using hbm_runtime.

The image encoder and fixed-prompt mask decoder are both quantized to RDK X5
`.bin` models. The prompt is baked into decoder ONNX, so board inference only
requires hbm_runtime, NumPy, and OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
from hbm_runtime import HB_HBMRuntime


@dataclass
class EfficientSAMConfig:
    """Runtime configuration for EfficientSAM-Tiny full-mask inference.

    Args:
        encoder_model_path: Path to the quantized EfficientSAM image encoder `.bin`.
        decoder_model_path: Path to the quantized fixed-prompt decoder `.bin`.
        input_size: Square input size used by the exported ONNX and `.bin` models.
    """

    encoder_model_path: str
    decoder_model_path: str
    input_size: int = 512


class EfficientSAMSegment:
    """EfficientSAM full-mask pipeline backed by two hbm_runtime models.

    The class owns the encoder and decoder runtimes and exposes the standard
    `pre_process`, `forward`, `post_process`, and `predict` stages used by RDK
    Model Zoo Python demos.
    """

    def __init__(self, config: EfficientSAMConfig) -> None:
        """Load encoder and decoder `.bin` models.

        Args:
            config: Runtime configuration with model paths.
        """

        self.config = config
        self.encoder = HB_HBMRuntime(config.encoder_model_path)
        self.decoder = HB_HBMRuntime(config.decoder_model_path)
        self.encoder_name = self.encoder.model_names[0]
        self.decoder_name = self.decoder.model_names[0]

    def set_scheduling_params(self, priority: int | None = None) -> None:
        """Set runtime priority when the hbm_runtime object exposes the API.

        RDK X5 has one BPU core, so this sample does not expose a BPU core
        selection argument.

        Args:
            priority: Optional runtime scheduling priority.
        """

        if priority is None:
            return
        for model in (self.encoder, self.decoder):
            if hasattr(model, "set_scheduling_params"):
                try:
                    model.set_scheduling_params(priority=priority)
                except TypeError:
                    pass

    def pre_process(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare the EfficientSAM encoder input tensor.

        Args:
            image: Input image in BGR HWC format.

        Returns:
            Dictionary containing `batched_images` in NCHW float32 RGB format.

        Raises:
            ValueError: If the input is not a three-channel image.
        """

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a BGR HWC image")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        return {"batched_images": np.ascontiguousarray(tensor, dtype=np.float32)}

    def forward(self, input_tensors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Run encoder and fixed-prompt decoder inference.

        Args:
            input_tensors: Encoder input dictionary from `pre_process`.

        Returns:
            Nested dictionaries containing raw encoder and decoder outputs.
        """

        encoder_outputs = self.encoder.run(input_tensors)
        image_embeddings = encoder_outputs[self.encoder_name]["image_embeddings"].astype(np.float32)
        decoder_outputs = self.decoder.run({"image_embeddings": image_embeddings})
        return {"encoder": encoder_outputs[self.encoder_name], "decoder": decoder_outputs[self.decoder_name]}

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray | float | int]:
        """Convert decoder logits to a full-resolution binary mask.

        Args:
            outputs: Raw outputs returned by `forward`.

        Returns:
            Dictionary with binary mask, selected IoU, selected mask index, and
            raw low-resolution masks.
        """

        low_res_masks = outputs["decoder"]["low_res_masks"].astype(np.float32)
        ious = outputs["decoder"]["iou_predictions"].astype(np.float32).reshape(-1)
        best_index = int(np.argmax(ious))
        logits = cv2.resize(low_res_masks[0, best_index], (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_LINEAR)
        mask = logits >= 0
        return {"mask": mask, "iou": float(ious[best_index]), "mask_index": best_index, "low_res_masks": low_res_masks}

    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        """Run the complete segmentation pipeline.

        Args:
            image: Input image in BGR HWC format.

        Returns:
            Postprocessed segmentation result dictionary.
        """

        return self.post_process(self.forward(self.pre_process(image)))

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray | float | int]:
        """Alias for `predict`."""

        return self.predict(image)


def draw_mask_result(image: np.ndarray, mask: np.ndarray, iou: float, mask_index: int) -> np.ndarray:
    """Draw a binary mask overlay and contour on the resized input image.

    Args:
        image: Original BGR input image.
        mask: Boolean full-resolution mask.
        iou: Predicted IoU of the selected mask.
        mask_index: Index of the selected decoder mask.

    Returns:
        BGR visualization image with mask overlay and text labels.
    """

    canvas = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    mask_bool = mask.astype(bool)
    color = np.zeros_like(canvas)
    color[:, :] = (0, 180, 0)
    blended = cv2.addWeighted(canvas, 0.45, color, 0.55, 0)
    canvas[mask_bool] = blended[mask_bool]
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    lines = ["EfficientSAM full mask: encoder.bin + decoder.bin", f"hbm_runtime dual model, mask={mask_index}, IoU={iou:.4f}"]
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    return canvas
