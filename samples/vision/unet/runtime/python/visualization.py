# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""VOC palette and coloring; not part of inference."""
import numpy as np

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
