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

"""Source display geometry and palette, separated from inference and file IO."""

from typing import Dict
import cv2
import numpy as np

BEV_PIXEL_SIZE = 0.25
BEV_CLASS_NAMES = (
    "background",
    "road",
    "walkway",
    "centerline",
    "static",
    "vehicle",
    "pedestrian",
)
BEV_PALETTE_BGR = np.asarray(
    [
        [255, 255, 255],
        [185, 185, 185],
        [167, 205, 232],
        [0, 215, 255],
        [182, 89, 155],
        [60, 76, 231],
        [219, 152, 52],
    ],
    dtype=np.uint8,
)


def _xy_to_pixel(x: float, y: float) -> tuple[int, int]:
    """Convert ego-local metric coordinates into the 256x256 LiDAR raster.

    Args:
        x: Forward coordinate in meters.
        y: Left coordinate in meters.

    Returns:
        OpenCV pixel coordinate as ``(column, row)``.
    """

    return int(round(y / BEV_PIXEL_SIZE + 128.0)), int(
        round(x / BEV_PIXEL_SIZE + 128.0)
    )


def _agent_polygon(state: np.ndarray) -> np.ndarray:
    """Convert one ``[x, y, heading, length, width]`` state to raster corners.

    Args:
        state: Agent state in ego-local metric coordinates.

    Returns:
        Four OpenCV polygon points as an int32 array.
    """

    x, y, heading, length, width = map(float, state)
    forward = np.array([np.cos(heading), np.sin(heading)]) * length / 2.0
    lateral = np.array([-np.sin(heading), np.cos(heading)]) * width / 2.0
    center = np.array([x, y])
    corners = [
        center + forward + lateral,
        center + forward - lateral,
        center - forward - lateral,
        center - forward + lateral,
    ]
    return np.asarray(
        [_xy_to_pixel(float(point[0]), float(point[1])) for point in corners],
        dtype=np.int32,
    )


def render_result(
    features: Dict[str, np.ndarray],
    result: Dict[str, np.ndarray],
    platform_name: str = "RDK S",
) -> np.ndarray:
    """Render camera, BEV semantic, LiDAR, trajectory, and agent visualization.

    Args:
        features: Original float input feature dictionary.
        result: Post-processed result returned by ``DiffusionDrive.predict``.
        platform_name: Board name shown in the visualization title.

    Returns:
        Owned BGR uint8 canvas; the caller controls saving.
    """

    camera = np.clip(features["camera"][0].transpose(1, 2, 0), 0.0, 1.0)
    camera_bgr = cv2.cvtColor(
        np.rint(camera * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    semantic = BEV_PALETTE_BGR[result["bev_labels"][0]]
    semantic = cv2.rotate(semantic, cv2.ROTATE_180)
    semantic = cv2.resize(semantic, (512, 256), interpolation=cv2.INTER_NEAREST)

    density = np.clip(features["lidar"][0, 0], 0.0, 1.0)
    gray = np.rint(255.0 * (1.0 - density)).astype(np.uint8)
    lidar = np.repeat(gray[..., None], 3, axis=-1)
    cv2.polylines(
        lidar,
        [_agent_polygon(np.array([0.0, 0.0, 0.0, 5.2, 2.0]))],
        True,
        (235, 99, 36),
        2,
    )
    for state in result["agent_states"][0, result["agent_mask"][0]]:
        cv2.polylines(lidar, [_agent_polygon(state)], True, (68, 68, 239), 2)
    trajectory = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), result["trajectory"][0, :, :2]], axis=0
    )
    points = np.asarray(
        [_xy_to_pixel(float(x), float(y)) for x, y in trajectory], dtype=np.int32
    )
    cv2.polylines(lidar, [points], False, (0, 122, 255), 3)
    for point in points[1:]:
        cv2.circle(lidar, tuple(point), 3, (0, 122, 255), -1)
    lidar = lidar[128:256]
    lidar = cv2.rotate(lidar, cv2.ROTATE_180)
    lidar = cv2.resize(lidar, (512, 256), interpolation=cv2.INTER_NEAREST)

    canvas = np.full((648, 1024, 3), (36, 29, 25), dtype=np.uint8)
    canvas[32:288] = camera_bgr
    canvas[320:576, :512] = semantic
    canvas[320:576, 512:] = lidar
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        f"{platform_name} DiffusionDrive - camera input",
        (12, 21),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Predicted BEV semantics",
        (12, 309),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    count = int(result["agent_mask"].sum())
    cv2.putText(
        canvas,
        f"LiDAR + trajectory (orange) + agents (red, count={count})",
        (524, 309),
        font,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    x_offset = 12
    for class_name, color in zip(BEV_CLASS_NAMES, BEV_PALETTE_BGR):
        cv2.rectangle(
            canvas,
            (x_offset, 591),
            (x_offset + 13, 604),
            tuple(int(value) for value in color),
            -1,
        )
        cv2.putText(
            canvas,
            class_name,
            (x_offset + 18, 603),
            font,
            0.38,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        x_offset += 139
    cv2.putText(
        canvas,
        "Forward is up; ego vehicle is blue.",
        (12, 634),
        font,
        0.45,
        (210, 205, 195),
        1,
        cv2.LINE_AA,
    )
    return canvas
