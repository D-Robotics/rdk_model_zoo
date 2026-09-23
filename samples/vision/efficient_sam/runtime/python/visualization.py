"""Pure visualization helpers for EfficientSAM output."""
from __future__ import annotations

import cv2
import numpy as np


def draw_mask_result(image: np.ndarray, mask: np.ndarray, iou: float, mask_index: int) -> np.ndarray:
    """Return a 512x512 BGR overlay; this function does not write files."""
    canvas = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (512, 512):
        raise ValueError(f"Expected a 512x512 mask, got {mask_bool.shape}.")
    color = np.zeros_like(canvas)
    color[:] = (0, 180, 0)
    blended = cv2.addWeighted(canvas, 0.45, color, 0.55, 0)
    canvas[mask_bool] = blended[mask_bool]
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    for row, line in enumerate(("EfficientSAM full mask: encoder + decoder", f"mask={mask_index}, IoU={float(iou):.4f}")):
        y = 28 + row * 28
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


__all__ = ["draw_mask_result"]
