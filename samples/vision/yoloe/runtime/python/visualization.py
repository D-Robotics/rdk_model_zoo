# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Image/label I/O and visualization, separate from numerical inference stages."""

from pathlib import Path
import hashlib
import cv2
import numpy as np

from samples.vision.yoloe.model.vocabulary import LABELS_SHA256


def load_inputs(image_path, label_path):
    image = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    raw = Path(label_path).expanduser().read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise ValueError(
            "Vocabulary checksum mismatch; the PF model requires its fixed ordered 4585 classes."
        )
    labels = raw.decode("utf-8").splitlines()
    if len(labels) != 4585:
        raise ValueError("Expected 4585 vocabulary entries.")
    return image, labels


def draw_result(image, result, labels, *, contours=True):
    canvas = image.copy()
    for box, score, cls, mask in zip(
        result.boxes, result.scores, result.class_ids, result.masks
    ):
        cls = int(cls)
        if not 0 <= cls < len(labels):
            raise ValueError("Class ID is outside the fixed vocabulary.")
        x1, y1, x2, y2 = box.astype(int)
        color = np.array(
            [(37 * cls + 50) % 256, (67 * cls + 80) % 256, (97 * cls + 110) % 256],
            dtype=np.uint8,
        )
        view = canvas if result.mask_layout == "full" else canvas[y1:y2, x1:x2]
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != view.shape[:2]:
            raise ValueError("Mask geometry differs from its declared layout.")
        view[selected] = (view[selected].astype(np.float32) * 0.6 + color * 0.4).astype(
            np.uint8
        )
        if contours and selected.size:
            curves, _ = cv2.findContours(
                selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(view, curves, -1, tuple(int(v) for v in color), 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2)
        cv2.putText(
            canvas,
            f"{labels[cls]} {float(score):.3f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            tuple(int(v) for v in color),
            1,
        )
    return canvas


def save_result(path, image, result, labels, *, contours=True):
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(destination), draw_result(image, result, labels, contours=contours)
    ):
        raise OSError(f"Cannot save result: {destination}")
