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

"""Small target-local geometry policies used by the OCR detector stage.

These helpers intentionally contain the only optional ``pyclipper`` import in
the pilot.  They preserve the audited source behavior: external contours are
expanded with the DB ratio, minimum-area rectangles are integer boxes, and X5
returns the original image for a zero-sized crop while S100 preserves the
source OpenCV call for that degenerate case.  The source also has different
multi-polygon conversion order, which is retained below.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def dilate_contours(
    contours: Iterable[np.ndarray],
    *,
    target: str,
    ratio_prime: float = 2.7,
) -> tuple[np.ndarray, ...]:
    """Expand contours with the target's source DB/pyclipper policy."""

    if target not in ("x5", "s100"):
        raise ValueError(f"Unsupported OCR geometry target {target!r}.")
    if not np.isfinite(ratio_prime) or ratio_prime < 0:
        raise ValueError("ratio_prime must be a finite non-negative number.")
    import cv2
    contours = tuple(contours)
    if not contours:
        return ()
    try:
        import pyclipper
    except ImportError as exc:  # pragma: no cover - exercised on SDK hosts
        raise RuntimeError(
            "pyclipper is required for OCR detection post-processing; install it "
            "for the execution environment."
        ) from exc

    expanded: list[np.ndarray] = []
    for contour in contours:
        polygon = np.asarray(contour)[:, 0, :]
        arc_length = cv2.arcLength(polygon, True)
        if arc_length == 0:
            continue
        distance = cv2.contourArea(polygon) * ratio_prime / arc_length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(polygon, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        solution = offset.Execute(distance)
        if target == "s100":
            # S100's source filters multi-polygons before conversion because
            # recent NumPy rejects ragged nested lists.
            if len(solution) != 1 or len(solution[0]) == 0:
                continue
            polygon_array = np.array(solution, dtype=np.int64)
        else:
            # X5's source converts first.  A ragged multi-polygon therefore
            # retains its observable NumPy ValueError behavior.
            polygon_array = np.array(solution)
        if polygon_array.size == 0 or len(polygon_array) != 1:
            continue
        expanded.append(polygon_array)
    return tuple(expanded)


def get_bounding_boxes(
    polygons: Iterable[np.ndarray],
    *,
    min_area: float = 100,
) -> tuple[np.ndarray, ...]:
    """Return source-compatible integer minimum-area rectangles."""

    if not np.isfinite(min_area) or min_area < 0:
        raise ValueError("min_area must be a finite non-negative number.")
    import cv2

    boxes: list[np.ndarray] = []
    for polygon in polygons:
        if cv2.contourArea(polygon) < min_area:
            continue
        rectangle = cv2.minAreaRect(polygon)
        boxes.append(cv2.boxPoints(rectangle).astype(np.int64))
    return tuple(boxes)


def crop_and_rotate_image(
    image: np.ndarray,
    box: np.ndarray,
    *,
    target: str,
) -> np.ndarray:
    """Perspective-crop one detected box with the target's edge policy."""

    if target not in ("x5", "s100"):
        raise ValueError(f"Unsupported OCR geometry target {target!r}.")
    import cv2

    rectangle = cv2.minAreaRect(box)
    box_points = cv2.boxPoints(rectangle).astype(np.intp)
    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    angle = rectangle[2]

    if target == "x5" and (width == 0 or height == 0):
        # This is an observable legacy X5 behavior and is retained as a local
        # policy.  DetectionResult takes an owned copy before returning it.
        return image
    src_points = box_points.astype("float32")
    dst_points = np.array(
        [
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, transform, (width, height))
    if angle >= 45:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


__all__ = ["crop_and_rotate_image", "dilate_contours", "get_bounding_boxes"]
