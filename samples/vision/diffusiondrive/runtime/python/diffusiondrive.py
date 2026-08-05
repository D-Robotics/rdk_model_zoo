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

"""DiffusionDrive trajectory-planning runtime wrapper for RDK S100P/S600.

The wrapper discovers tensor names, shapes, data types, and quantization scales
from the HBM model. Float32 NAVSIM features are quantized before inference, and
fixed-point outputs are dequantized before trajectory and BEV post-processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import hbm_runtime
import numpy as np


INPUT_NAMES = ("camera", "lidar", "status", "noise")
OUTPUT_NAMES = ("trajectory", "agent_states", "agent_labels", "bev_semantic_map")
BEV_PIXEL_SIZE = 0.25
BEV_CLASS_NAMES = ("background", "road", "walkway", "centerline", "static", "vehicle", "pedestrian")
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


@dataclass
class DiffusionDriveConfig:
    """Configure the DiffusionDrive RDK S runtime.

    Args:
        model_path: Path to the compiled DiffusionDrive HBM model.
        agent_score_threshold: Minimum sigmoid score for predicted agents.
    """

    model_path: str
    agent_score_threshold: float = 0.5


class DiffusionDrive:
    """Run four-input DiffusionDrive planning inference with ``hbm_runtime``.

    Args:
        config: Model path and agent filtering configuration.

    Attributes:
        cfg: Runtime configuration supplied during construction.
        model_name: Public model name discovered from the HBM file.
    """

    def __init__(self, config: DiffusionDriveConfig):
        """Load the HBM model and validate its public tensor interface.

        Args:
            config: Model path and post-processing configuration.

        Raises:
            ValueError: If the HBM tensor interface does not match this sample.
        """

        self.cfg = config
        self.model = hbm_runtime.HB_HBMRuntime(config.model_path)
        self.model_name = self.model.model_names[0]
        input_names = tuple(self.model.input_names[self.model_name])
        output_names = tuple(self.model.output_names[self.model_name])
        if input_names != INPUT_NAMES:
            raise ValueError(f"Expected inputs {INPUT_NAMES}, got {input_names}")
        if set(output_names) != set(OUTPUT_NAMES):
            raise ValueError(f"Expected outputs {OUTPUT_NAMES}, got {output_names}")

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set optional runtime priority and BPU core affinity.

        Args:
            priority: Scheduling priority in the range supported by the runtime.
            bpu_cores: BPU core indexes used for this model.

        Returns:
            None.
        """

        kwargs = {}
        if priority is not None:
            kwargs["priority"] = {self.model_name: priority}
        if bpu_cores is not None:
            kwargs["bpu_cores"] = {self.model_name: bpu_cores}
        if kwargs:
            self.model.set_scheduling_params(**kwargs)

    @staticmethod
    def _numpy_dtype(runtime_dtype) -> np.dtype:
        """Map an hbm_runtime tensor data type to a NumPy data type.

        Args:
            runtime_dtype: Data type descriptor reported by ``hbm_runtime``.

        Returns:
            Corresponding NumPy data type.

        Raises:
            ValueError: If the runtime data type is unsupported.
        """

        name = str(runtime_dtype).split(".")[-1]
        mapping = {
            "S8": np.dtype(np.int8),
            "U8": np.dtype(np.uint8),
            "S16": np.dtype(np.int16),
            "U16": np.dtype(np.uint16),
            "S32": np.dtype(np.int32),
            "U32": np.dtype(np.uint32),
            "F16": np.dtype(np.float16),
            "F32": np.dtype(np.float32),
        }
        if name not in mapping:
            raise ValueError(f"Unsupported runtime dtype: {runtime_dtype}")
        return mapping[name]

    @staticmethod
    def _quantize(tensor: np.ndarray, quant, dtype: np.dtype) -> np.ndarray:
        """Quantize one float tensor using HBM scale and zero-point metadata.

        Args:
            tensor: Float input tensor.
            quant: Quantization metadata reported by ``hbm_runtime``.
            dtype: Integer or float destination data type.

        Returns:
            Contiguous tensor ready for BPU inference.

        Raises:
            ValueError: If the input uses unsupported per-channel quantization.
        """

        scales = np.asarray(quant.scale, dtype=np.float32)
        zero_points = np.asarray(quant.zero_point, dtype=np.float32)
        if scales.size == 0:
            return np.ascontiguousarray(tensor, dtype=dtype)
        if scales.size != 1:
            raise ValueError("This sample currently expects per-tensor input quantization")
        zero_point = float(zero_points[0]) if zero_points.size else 0.0
        values = np.rint(tensor.astype(np.float32) / float(scales[0]) + zero_point)
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            values = np.clip(values, limits.min, limits.max)
        return np.ascontiguousarray(values, dtype=dtype)

    @staticmethod
    def _dequantize(tensor: np.ndarray, quant) -> np.ndarray:
        """Convert one fixed-point output tensor to float32.

        Args:
            tensor: Raw output tensor returned by ``hbm_runtime``.
            quant: Quantization metadata reported by ``hbm_runtime``.

        Returns:
            Dequantized float32 tensor.
        """

        scales = np.asarray(quant.scale, dtype=np.float32)
        zero_points = np.asarray(quant.zero_point, dtype=np.float32)
        if scales.size == 0:
            return tensor.astype(np.float32, copy=False)
        values = tensor.astype(np.float32)
        if scales.size == 1:
            zero_point = float(zero_points[0]) if zero_points.size else 0.0
            return (values - zero_point) * float(scales[0])
        axis = int(quant.axis)
        shape = [1] * values.ndim
        shape[axis] = scales.size
        zeros = zero_points if zero_points.size else np.zeros_like(scales)
        return (values - zeros.reshape(shape)) * scales.reshape(shape)

    def pre_process(self, features: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """Validate and quantize NAVSIM feature tensors.

        Args:
            features: Float32 tensors named ``camera``, ``lidar``, ``status``,
                and ``noise``.

        Returns:
            Nested dictionary accepted directly by ``hbm_runtime.run``.

        Raises:
            ValueError: If a required tensor is missing or has the wrong shape.
        """

        prepared = {}
        shapes = self.model.input_shapes[self.model_name]
        dtypes = self.model.input_dtypes[self.model_name]
        quants = self.model.input_quants[self.model_name]
        for name in INPUT_NAMES:
            if name not in features:
                raise ValueError(f"Missing input tensor: {name}")
            tensor = np.asarray(features[name], dtype=np.float32)
            expected_shape = tuple(shapes[name])
            if tensor.shape != expected_shape:
                raise ValueError(f"{name}: expected {expected_shape}, got {tensor.shape}")
            prepared[name] = self._quantize(tensor, quants[name], self._numpy_dtype(dtypes[name]))
        return {self.model_name: prepared}

    def forward(
        self,
        input_tensors: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Execute one DiffusionDrive inference on the BPU.

        Args:
            input_tensors: Nested input dictionary returned by ``pre_process``.

        Returns:
            Raw nested outputs returned by ``hbm_runtime``.
        """

        return self.model.run(input_tensors)

    def post_process(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Dequantize trajectory, agent, and BEV semantic outputs.

        Args:
            outputs: Raw nested outputs returned by ``forward``.

        Returns:
            Dictionary containing float trajectory, agent states/scores, BEV
            logits, and the per-pixel BEV class map.
        """

        raw = outputs[self.model_name]
        quants = self.model.output_quants[self.model_name]
        decoded = {name: self._dequantize(raw[name], quants[name]) for name in OUTPUT_NAMES}
        logits = np.clip(decoded["agent_labels"], -60.0, 60.0)
        agent_scores = 1.0 / (1.0 + np.exp(-logits))
        agent_mask = agent_scores >= self.cfg.agent_score_threshold
        bev_logits = decoded["bev_semantic_map"]
        return {
            "trajectory": decoded["trajectory"],
            "agent_states": decoded["agent_states"],
            "agent_scores": agent_scores,
            "agent_mask": agent_mask,
            "bev_logits": bev_logits,
            "bev_labels": np.argmax(bev_logits, axis=1).astype(np.uint8),
        }

    def predict(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run preprocessing, BPU inference, and post-processing.

        Args:
            features: Float32 NAVSIM feature tensors.

        Returns:
            Decoded trajectory, agents, and BEV semantics.
        """

        return self.post_process(self.forward(self.pre_process(features)))

    def __call__(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run the same inference path as ``predict``.

        Args:
            features: Float32 NAVSIM feature tensors.

        Returns:
            Decoded trajectory, agents, and BEV semantics.
        """

        return self.predict(features)


def _xy_to_pixel(x: float, y: float) -> tuple[int, int]:
    """Convert ego-local metric coordinates into the 256x256 LiDAR raster.

    Args:
        x: Forward coordinate in meters.
        y: Left coordinate in meters.

    Returns:
        OpenCV pixel coordinate as ``(column, row)``.
    """

    return int(round(y / BEV_PIXEL_SIZE + 128.0)), int(round(x / BEV_PIXEL_SIZE + 128.0))


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
    corners = [center + forward + lateral, center + forward - lateral, center - forward - lateral, center - forward + lateral]
    return np.asarray([_xy_to_pixel(float(point[0]), float(point[1])) for point in corners], dtype=np.int32)


def render_result(
    features: Dict[str, np.ndarray],
    result: Dict[str, np.ndarray],
    output_path: str,
    platform_name: str = "RDK S",
) -> None:
    """Save camera, BEV semantic, LiDAR, trajectory, and agent visualization.

    Args:
        features: Original float input feature dictionary.
        result: Post-processed result returned by ``DiffusionDrive.predict``.
        output_path: Destination PNG or JPEG path.
        platform_name: Board name shown in the visualization title.

    Returns:
        None.

    Raises:
        RuntimeError: If OpenCV cannot write the output image.
    """

    camera = np.clip(features["camera"][0].transpose(1, 2, 0), 0.0, 1.0)
    camera_bgr = cv2.cvtColor(np.rint(camera * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    semantic = BEV_PALETTE_BGR[result["bev_labels"][0]]
    semantic = cv2.rotate(semantic, cv2.ROTATE_180)
    semantic = cv2.resize(semantic, (512, 256), interpolation=cv2.INTER_NEAREST)

    density = np.clip(features["lidar"][0, 0], 0.0, 1.0)
    gray = np.rint(255.0 * (1.0 - density)).astype(np.uint8)
    lidar = np.repeat(gray[..., None], 3, axis=-1)
    cv2.polylines(lidar, [_agent_polygon(np.array([0.0, 0.0, 0.0, 5.2, 2.0]))], True, (235, 99, 36), 2)
    for state in result["agent_states"][0, result["agent_mask"][0]]:
        cv2.polylines(lidar, [_agent_polygon(state)], True, (68, 68, 239), 2)
    trajectory = np.concatenate([np.zeros((1, 2), dtype=np.float32), result["trajectory"][0, :, :2]], axis=0)
    points = np.asarray([_xy_to_pixel(float(x), float(y)) for x, y in trajectory], dtype=np.int32)
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
    cv2.putText(canvas, "Predicted BEV semantics", (12, 309), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    count = int(result["agent_mask"].sum())
    cv2.putText(canvas, f"LiDAR + trajectory (orange) + agents (red, count={count})", (524, 309), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    x_offset = 12
    for class_name, color in zip(BEV_CLASS_NAMES, BEV_PALETTE_BGR):
        cv2.rectangle(canvas, (x_offset, 591), (x_offset + 13, 604), tuple(int(value) for value in color), -1)
        cv2.putText(canvas, class_name, (x_offset + 18, 603), font, 0.38, (225, 225, 225), 1, cv2.LINE_AA)
        x_offset += 139
    cv2.putText(canvas, "Forward is up; ego vehicle is blue.", (12, 634), font, 0.45, (210, 205, 195), 1, cv2.LINE_AA)
    if not cv2.imwrite(output_path, canvas):
        raise RuntimeError(f"Failed to save visualization: {output_path}")
