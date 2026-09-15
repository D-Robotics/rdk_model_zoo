# Copyright (c) 2025 D-Robotics Corporation
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

"""Adapt NV12 planes to the input protocol a platform's model expects.

RDK X5 `.bin` models are compiled with one packed NV12 input tensor. RDK S
`.hbm` models are compiled with the luma and chroma planes bound as two
separate tensors. The rest of the pipeline is identical, so the difference is
confined to this adapter.

The adapter is deliberately strict. It reads the real input tensor metadata
reported by the runtime, derives the input geometry from it, cross-checks the
element counts it is about to bind, and refuses anything it cannot describe
exactly. It never guesses a protocol from the platform name alone.

Typical Usage:
    >>> adapter = Nv12InputAdapter.from_metadata(
    ...     profile, "model", ["y", "uv"],
    ...     {"y": (1, 640, 640, 1), "uv": (1, 320, 320, 2)})
    >>> sorted(adapter.build(y_plane, uv_plane)["model"])
    ['uv', 'y']
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from yolo_platform import PlatformProfile

#: Channel dimension position of a planar NCHW tensor.
_LAYOUT_NCHW = "NCHW"

#: Channel dimension position of an interleaved NHWC tensor.
_LAYOUT_NHWC = "NHWC"

#: Named input layouts the adapter recognises, keyed by the channel-count set.
_KNOWN_LAYOUTS = {
    _LAYOUT_NCHW: 3,
    _LAYOUT_NHWC: 1,
}


class UnsupportedInputError(ValueError):
    """Raised when model input metadata cannot be described exactly."""


@dataclass(frozen=True)
class InputGeometry:
    """Describe the geometry of one model input tensor.

    Attributes:
        height: Input height in pixels.
        width: Input width in pixels.
        layout: `"NCHW"`, `"NHWC"`, or `"flat"` when the tensor is reported as
            a two-dimensional buffer.
        shape: The tensor shape as reported by the runtime.
    """

    height: int
    width: int
    layout: str
    shape: Tuple[int, ...]

    @property
    def pixels(self) -> int:
        """Return the number of luma samples in one frame."""
        return self.height * self.width


def _int_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    """Normalise a runtime shape into a tuple of Python integers."""
    return tuple(int(dim) for dim in shape)


def _element_count(shape: Sequence[int]) -> int:
    """Return the total number of elements of a tensor shape."""
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _infer_plane_geometry(shape: Tuple[int, ...],
                          default_hw: Optional[Tuple[int, int]],
                          label: str) -> InputGeometry:
    """Derive the pixel geometry of a single-plane NV12 input.

    Args:
        shape: Tensor shape as reported by the runtime.
        default_hw: `(height, width)` assumed when the shape is a flat buffer
            whose element count matches it.
        label: Input name, used in error messages.

    Returns:
        The inferred `InputGeometry`.

    Raises:
        UnsupportedInputError: If the geometry cannot be derived.
    """
    if len(shape) == 4:
        # NCHW: (N, C, H, W) with a single channel.
        if shape[1] == 1:
            return InputGeometry(shape[2], shape[3], _LAYOUT_NCHW, shape)
        # NHWC: (N, H, W, C) with a single channel.
        if shape[3] == 1:
            return InputGeometry(shape[1], shape[2], _LAYOUT_NHWC, shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports shape {shape}, which is not a single-plane "
            f"NV12 tensor. Expected (1, 1, H, W) or (1, H, W, 1).")
    if len(shape) == 2:
        count = _element_count(shape)
        if default_hw is not None and default_hw[0] * default_hw[1] == count:
            return InputGeometry(default_hw[0], default_hw[1], "flat", shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports flat shape {shape} ({count} elements), so "
            f"its height and width cannot be derived. Pass --input-shape HxW "
            f"matching the compiled model.")
    if len(shape) == 3 and shape[2] == 1:
        return InputGeometry(shape[0], shape[1], _LAYOUT_NHWC, shape)
    raise UnsupportedInputError(
        f"Input {label!r} reports shape {shape}, which is not a supported "
        f"single-plane NV12 tensor layout.")


def _infer_packed_geometry(shape: Tuple[int, ...],
                           default_hw: Optional[Tuple[int, int]],
                           label: str) -> InputGeometry:
    """Derive the pixel geometry of a packed NV12 input.

    A packed NV12 tensor carries one luma plane plus one interleaved chroma
    plane, so its element count is `H * W * 3 / 2`. Some runtimes report the
    same buffer as an `H x W x 3` view instead.

    Args:
        shape: Tensor shape as reported by the runtime.
        default_hw: `(height, width)` assumed when the shape carries no usable
            spatial dimensions.
        label: Input name, used in error messages.

    Returns:
        The inferred `InputGeometry`.

    Raises:
        UnsupportedInputError: If the geometry cannot be derived.
    """
    if len(shape) == 4:
        if shape[1] == 3:
            return InputGeometry(shape[2], shape[3], _LAYOUT_NCHW, shape)
        if shape[3] in (3, 4):
            return InputGeometry(shape[1], shape[2], _LAYOUT_NHWC, shape)
        raise UnsupportedInputError(
            f"Input {label!r} reports shape {shape}, which is not a packed NV12 "
            f"tensor. Expected (1, 3, H, W) or (1, H, W, 3).")
    count = _element_count(shape)
    if default_hw is not None:
        expected = default_hw[0] * default_hw[1] * 3 // 2
        if expected == count:
            return InputGeometry(default_hw[0], default_hw[1], "flat", shape)
    raise UnsupportedInputError(
        f"Input {label!r} reports shape {shape} ({count} elements), which does "
        f"not describe a square packed NV12 frame. Pass --input-shape HxW.")


def _validate_square(geometry: InputGeometry, label: str) -> None:
    """Reject non-square inputs, which the shared decoder cannot handle.

    The DFL decoder builds an anchor grid from a single spatial size, so a
    rectangular input would silently decode against the wrong grid. It is
    rejected rather than approximated.

    Args:
        geometry: Geometry to validate.
        label: Input name, used in error messages.

    Raises:
        UnsupportedInputError: If the input is not square.
    """
    if min(geometry.height, geometry.width) <= 0 or geometry.height % 2 or geometry.width % 2:
        raise UnsupportedInputError("NV12 dimensions must be positive and even.")
    if geometry.height != geometry.width:
        raise UnsupportedInputError(
            f"Input {label!r} is {geometry.height}x{geometry.width}. This sample "
            f"supports square inputs only, because the DFL anchor grid is built "
            f"from a single spatial size.")


@dataclass
class Nv12InputAdapter:
    """Bind NV12 planes to a model input in the platform's protocol.

    Attributes:
        profile: Platform whose protocol is used.
        model_name: Model name used as the outer key of the input dictionary.
        input_names: Input tensor names, in runtime order.
        geometry: Geometry inferred from the first input tensor.
        consumed_elements: Total element count the adapter binds.
    """

    profile: PlatformProfile
    model_name: str
    input_names: List[str]
    geometry: InputGeometry
    consumed_elements: int

    @property
    def input_height(self) -> int:
        """Return the model input height in pixels."""
        return self.geometry.height

    @property
    def input_width(self) -> int:
        """Return the model input width in pixels."""
        return self.geometry.width

    @property
    def is_packed(self) -> bool:
        """Return True when NV12 is bound as one packed tensor."""
        return self.profile.is_packed_input

    def expected_anchor_sizes(self, strides: Sequence[int]) -> List[int]:
        """Return the feature-map grid size of each detection scale.

        Args:
            strides: Downsampling stride of each detection scale.

        Returns:
            The grid size of each scale, derived from the input height.
        """
        if len(strides) != 3 or any(int(s) <= 0 or self.geometry.height % int(s) for s in strides):
            raise UnsupportedInputError("Three positive strides must divide the model input size.")
        return [self.geometry.height // int(stride) for stride in strides]

    def describe(self) -> str:
        """Return a one-line description of the bound input protocol.

        Returns:
            A human-readable summary used in logs and in `--list-models`.
        """
        return (f"{self.profile.key}: {self.profile.input_protocol} NV12, "
                f"{self.geometry.height}x{self.geometry.width}, "
                f"layout {self.geometry.layout}")

    @classmethod
    def from_metadata(cls,
                      profile: PlatformProfile,
                      model_name: str,
                      input_names: Sequence[str],
                      input_shapes: Dict[str, Sequence[int]],
                      input_shape_override: Optional[Tuple[int, int]] = None
                      ) -> "Nv12InputAdapter":
        """Build an adapter from the input metadata a runtime reports.

        Args:
            profile: Platform whose protocol is used.
            model_name: Model name used as the outer key of the input dict.
            input_names: Input tensor names, in the order the runtime reports.
            input_shapes: Tensor shapes keyed by input name.
            input_shape_override: Explicit `(height, width)` used when the
                reported shapes carry no usable spatial metadata.

        Returns:
            A validated `Nv12InputAdapter`.

        Raises:
            UnsupportedInputError: If the input count, tensor layout or
                element count does not match the platform protocol.
        """
        names = [str(name) for name in input_names]
        expected_count = 1 if profile.is_packed_input else 2
        if len(names) != expected_count:
            raise UnsupportedInputError(
                f"{profile.key} models take {expected_count} NV12 input "
                f"tensor(s), but {model_name!r} reports {len(names)}: "
                f"{', '.join(names) or '(none)'}. The model was compiled for a "
                f"different input protocol.")
        for name in names:
            if name not in input_shapes:
                raise UnsupportedInputError(
                    f"Runtime reports input {name!r} for {model_name!r} without "
                    f"a shape.")

        default_hw = input_shape_override
        primary = _int_tuple(input_shapes[names[0]])
        if any(int(d) <= 0 for name in names for d in input_shapes[name]):
            raise UnsupportedInputError("Input shapes must be positive and static.")
        if len(primary) == 4 and primary[0] != 1:
            raise UnsupportedInputError("Only batch 1 NV12 inputs are supported.")

        if profile.is_packed_input:
            geometry = _infer_packed_geometry(primary, default_hw, names[0])
            _validate_square(geometry, names[0])
            if input_shape_override and tuple(input_shape_override) != (geometry.height, geometry.width):
                raise UnsupportedInputError("--input-shape conflicts with model metadata.")
            expected_payload = geometry.pixels * 3 // 2
            declared = _element_count(primary)
            if declared not in (expected_payload, geometry.pixels * 3):
                raise UnsupportedInputError(
                    f"Input {names[0]!r} declares {declared} elements, but a "
                    f"{geometry.height}x{geometry.width} packed NV12 frame needs "
                    f"{expected_payload}. The model input resolution does not "
                    f"match the shape this sample derived.")
            return cls(profile, model_name, names, geometry, expected_payload)

        geometry = _infer_plane_geometry(primary, default_hw, names[0])
        _validate_square(geometry, names[0])
        if input_shape_override and tuple(input_shape_override) != (geometry.height, geometry.width):
            raise UnsupportedInputError("--input-shape conflicts with model metadata.")
        if primary != (1, geometry.height, geometry.width, 1):
            raise UnsupportedInputError("Split NV12 requires NHWC Y (1,H,W,1).")
        if tuple(input_shapes[names[1]]) != (1, geometry.height//2, geometry.width//2, 2):
            raise UnsupportedInputError("Split NV12 requires UV (1,H/2,W/2,2).")
        luma_shape = _int_tuple(input_shapes[names[1]])
        luma_elements = _element_count(primary)
        chroma_elements = _element_count(luma_shape)
        if luma_elements != geometry.pixels:
            raise UnsupportedInputError(
                f"Input {names[0]!r} declares {luma_elements} elements, but a "
                f"{geometry.height}x{geometry.width} luma plane needs "
                f"{geometry.pixels}.")
        if chroma_elements * 2 != luma_elements:
            raise UnsupportedInputError(
                f"Input {names[1]!r} declares {chroma_elements} elements, but "
                f"NV12 chroma must hold exactly half the luma samples "
                f"({luma_elements // 2}).")
        return cls(profile, model_name, names, geometry, luma_elements + chroma_elements)

    def build(self, y_plane: np.ndarray, uv_plane: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Bind NV12 planes into the runtime input dictionary.

        Args:
            y_plane: Luma plane with `H * W` elements.
            uv_plane: Interleaved chroma plane with `H * W / 2` elements.

        Returns:
            A nested input dictionary in the form `{model_name: {name: tensor}}`.

        Raises:
            UnsupportedInputError: If the supplied planes do not match the
                geometry the adapter validated at construction time.
        """
        pixels = self.geometry.pixels
        if y_plane.size != pixels:
            raise UnsupportedInputError(
                f"Luma plane holds {y_plane.size} elements, expected {pixels} "
                f"for a {self.geometry.height}x{self.geometry.width} input.")
        if uv_plane.size * 2 != pixels:
            raise UnsupportedInputError(
                f"Chroma plane holds {uv_plane.size} elements, expected "
                f"{pixels // 2}.")

        if self.is_packed:
            packed = np.concatenate(
                [y_plane.reshape(-1), uv_plane.reshape(-1)]).astype(np.uint8)
            return {self.model_name: {self.input_names[0]: packed}}

        return {
            self.model_name: {
                self.input_names[0]: np.ascontiguousarray(y_plane, dtype=np.uint8).reshape(1, self.input_height, self.input_width, 1),
                self.input_names[1]: np.ascontiguousarray(uv_plane, dtype=np.uint8).reshape(1, self.input_height//2, self.input_width//2, 2),
            }
        }
