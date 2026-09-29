# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source Cityscapes colors and three-panel rendering, separate from inference."""
import cv2
import numpy as np


CITYSCAPES_PALETTE_BGR = np.array([
    [128,  64, 128],  # 0  road
    [232,  35, 244],  # 1  sidewalk
    [ 70,  70,  70],  # 2  building
    [156, 102, 102],  # 3  wall
    [153, 153, 190],  # 4  fence
    [153, 153, 153],  # 5  pole
    [ 30, 170, 250],  # 6  traffic light
    [  0, 220, 220],  # 7  traffic sign
    [ 35, 142, 107],  # 8  vegetation
    [152, 251, 152],  # 9  terrain
    [180, 130,  70],  # 10 sky
    [ 60,  20, 220],  # 11 person
    [  0,   0, 255],  # 12 rider
    [142,   0,   0],  # 13 car
    [100,  60,   0],  # 14 truck
    [ 70,   0,   0],  # 15 bus
    [100,  80,   0],  # 16 train
    [230,   0,   0],  # 17 motorcycle
    [ 32,  11, 119],  # 18 bicycle
], dtype=np.uint8)


CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]


def colorize(seg: np.ndarray) -> np.ndarray:
    """Map class indices to BGR colors using the Cityscapes palette."""
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(len(CITYSCAPES_PALETTE_BGR)):
        out[seg == cid] = CITYSCAPES_PALETTE_BGR[cid]
    return out


def draw_legend(canvas: np.ndarray, cls_ids: list) -> np.ndarray:
    """Overlay a small class legend on the top-right corner of canvas."""
    box, pad = 18, 6
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    leg_h = (box + pad) * len(cls_ids) + pad
    leg_w = 155
    leg = np.full((leg_h, leg_w, 3), 30, dtype=np.uint8)
    for i, cid in enumerate(cls_ids):
        y0 = pad + i * (box + pad)
        c = [int(x) for x in CITYSCAPES_PALETTE_BGR[cid]]
        cv2.rectangle(leg, (pad, y0), (pad + box, y0 + box), c, -1)
        cv2.putText(leg, CITYSCAPES_CLASS_NAMES[cid], (pad + box + 4, y0 + box - 3),
                    font, fs, (220, 220, 220), th)
    h, w = canvas.shape[:2]
    canvas[4: 4 + leg_h, w - leg_w - 4: w - 4] = leg
    return canvas


def render_result(bgr: np.ndarray, seg: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Produce a 3-panel result image: Original | Overlay | Segmentation.

    Args:
        bgr: Original BGR input image (any size).
        seg: Segmentation map (input_height, input_width) int32.

    Returns:
        Concatenated result image (H+header, 3*W+dividers, 3) uint8.
    """
    w, h = 1024, 512

    seg_color = colorize(seg)
    orig_rsz = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(orig_rsz, 1 - alpha, seg_color, alpha, 0)

    unique = sorted(np.unique(seg).tolist())
    valid = [c for c in unique if 0 <= c < len(CITYSCAPES_CLASS_NAMES)]
    overlay = draw_legend(overlay, valid)

    div = np.full((h, 3, 3), 60, dtype=np.uint8)
    panel = np.hstack([orig_rsz, div, overlay, div, seg_color])

    hdr = np.full((36, panel.shape[1], 3), 35, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(hdr, "Original", (10, 25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, f"Overlay  alpha={alpha:.2f}", (w + 13, 25), font, 0.65, (200, 200, 200), 1)
    cv2.putText(hdr, "Segmentation", (2 * w + 16, 25), font, 0.65, (200, 200, 200), 1)
    return np.vstack([hdr, panel])
