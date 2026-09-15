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

"""Bridge to the board runtime, kept lazy and dependency-free.

`hbm_runtime` ships with the board system image and is not installed on a
development host. Importing it at module scope would make `--help`,
`--list-models` and download dry runs fail on a host that has no board runtime,
so every access to it goes through this module and happens only when a model is
actually loaded.

The module also holds the shared parameter checks. The published models all use
16 DFL bins and 17 COCO keypoints, and the decoders reshape against those exact
counts, so a model compiled with different values is rejected instead of being
silently decoded against the wrong shape.

Typical Usage:
    >>> require_dfl_bins(16)
    >>> require_dfl_bins(8)
    Traceback (most recent call last):
    ...
    ValueError: ...
"""

from typing import Optional, Tuple

from yolo_assets import TASK_CLS
from yolo_input import Nv12InputAdapter
from yolo_platform import PlatformProfile

#: Number of DFL regression bins every published model uses.
SUPPORTED_DFL_BINS = 16

#: Number of keypoints every published pose model uses.
SUPPORTED_KEYPOINTS = 17


class BoardRuntimeUnavailableError(RuntimeError):
    """Raised when the board runtime is required but not installed."""


def load_hbm_runtime():
    """Import and return the board runtime module.

    Returns:
        The imported `hbm_runtime` module.

    Raises:
        BoardRuntimeUnavailableError: If the module is not installed. The
            message names the missing package and never triggers an install.
    """
    try:
        import hbm_runtime  # noqa: PLC0415 - deliberately lazy
    except ImportError as exc:
        raise BoardRuntimeUnavailableError(
            "hbm_runtime is not installed. It is provided by the RDK system "
            "image and is required only for on-board inference. Inspecting "
            "model names, printing --help and download dry runs work without "
            "it.") from exc
    return hbm_runtime


def require_dfl_bins(reg: int) -> int:
    """Validate the DFL bin count against what the decoder supports.

    Args:
        reg: Requested number of DFL regression bins.

    Returns:
        The validated bin count.

    Raises:
        ValueError: If the count is not the supported value.
    """
    value = int(reg)
    if value != SUPPORTED_DFL_BINS:
        raise ValueError(
            f"Unsupported DFL bin count --reg {value}. The decoder reshapes the "
            f"box output against {SUPPORTED_DFL_BINS} bins; models compiled with "
            f"a different count are not supported by this sample.")
    return value


def require_keypoints(nkpt: int) -> int:
    """Validate the keypoint count against what the decoder supports.

    Args:
        nkpt: Requested number of keypoints.

    Returns:
        The validated keypoint count.

    Raises:
        ValueError: If the count is not the supported value.
    """
    value = int(nkpt)
    if value != SUPPORTED_KEYPOINTS:
        raise ValueError(
            f"Unsupported keypoint count --nkpt {value}. The decoder reshapes "
            f"the keypoint output against {SUPPORTED_KEYPOINTS} COCO keypoints; "
            f"models trained with a different skeleton are not supported by "
            f"this sample.")
    return value


def require_square_grid(anchor_sizes, input_height: int) -> None:
    """Validate that computed anchor grids are square and positive.

    Args:
        anchor_sizes: Grid size of each detection scale.
        input_height: Model input height in pixels.

    Raises:
        ValueError: If a scale yields a non-square or empty grid.
    """
    for size in anchor_sizes:
        if int(size) <= 0:
            raise ValueError(
                f"Detection scale grid {size} is empty for an input height of "
                f"{input_height}. The model input resolution is smaller than "
                f"one of the configured strides.")


def default_resize_type(profile: PlatformProfile, task: str) -> int:
    """Return the platform's documented default resize policy for a task.

    The two published trees resized classification inputs differently: the X5
    tree ran classification with letterbox resizing, the S tree with a direct
    stretch. Detection, segmentation and pose used letterbox resizing on both.

    Args:
        profile: Platform supplying the default.
        task: Task name.

    Returns:
        `0` for stretch resize, `1` for letterbox resize.
    """
    return profile.cls_resize_type if task == TASK_CLS else 1


def default_nms_thres(profile: PlatformProfile, task: str) -> Optional[float]:
    """Return the platform's documented default NMS threshold for a task.

    Args:
        profile: Platform supplying the default.
        task: Task name.

    Returns:
        The IoU threshold, or `None` for tasks that do not run NMS.
    """
    if task == TASK_CLS:
        return None
    return profile.nms_thres


def open_model(config,
               profile: Optional[PlatformProfile],
               input_shape: Optional[Tuple[int, int]] = None):
    """Load a compiled model and describe its NV12 input protocol.

    Args:
        config: Wrapper configuration carrying at least `model_path`.
        profile: Platform profile describing the expected input protocol.
        input_shape: Explicit `(height, width)` override used when the runtime
            does not report usable spatial metadata.

    Returns:
        A tuple of the runtime model handle and the validated
        `Nv12InputAdapter`.

    Raises:
        BoardRuntimeUnavailableError: If the board runtime is missing.
        UnsupportedInputError: If the reported inputs do not match the
            platform protocol.
    """
    if profile is None:
        raise ValueError(
            "A platform profile is required to open a model. Select one with "
            "--platform or run on a supported board.")
    hbm_runtime = load_hbm_runtime()
    model = hbm_runtime.HB_HBMRuntime(config.model_path)
    model_name = model.model_names[0]
    adapter = Nv12InputAdapter.from_metadata(
        profile,
        model_name,
        model.input_names[model_name],
        model.input_shapes[model_name],
        input_shape_override=input_shape,
    )
    return model, adapter
