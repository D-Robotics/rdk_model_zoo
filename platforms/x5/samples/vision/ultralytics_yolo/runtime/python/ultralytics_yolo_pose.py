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
"""RDK X5 Ultralytics YOLO pose runtime wrapper (compatibility names).

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/runtime/python/yolo_pose.py`. This module keeps
the RDK X5 module, class and configuration names importable, so existing RDK X5
code and documentation keep working, without a second copy of the pipeline.

# The X5 wrapper documented an NMS default of `0.70` and a `classes_num`
# of 1. A pose model always predicts one class, which is what the merged
# wrapper assumes, so `classes_num` is kept for signature compatibility.
The runtime behaviour is the canonical behaviour; only the names and the
platform-bound defaults below are reinstated here.

Typical Usage:
    >>> from yolo_pose import UltralyticsYOLOPoseConfig, UltralyticsYOLOPose
    >>> net = UltralyticsYOLOPose(UltralyticsYOLOPoseConfig(model_path="yolo11n_pose_bayese_640x640_nv12.bin"))
    >>> results = net(img)
"""

import os
import sys


def _find_sample_root() -> str:
    """Locate the canonical Ultralytics YOLO sample above this file.

    The platform trees are distributions of the merged sample, so the
    canonical copy always sits at `<repo>/samples/vision/ultralytics_yolo`.
    Walking up to it instead of counting `..` keeps this working when the
    sample is moved, and turns a missing canonical sample into an explicit
    error instead of a confusing import failure.

    A directory only counts as the canonical sample when it carries the shared
    downloader, because every platform tree ends in the same
    `samples/vision/ultralytics_yolo` tail and would otherwise match first.

    Returns:
        Absolute path of the canonical sample directory.

    Raises:
        ImportError: If no parent directory holds the canonical sample.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(current, "samples", "vision",
                                 "ultralytics_yolo")
        if os.path.isfile(os.path.join(candidate, "runtime", "python",
                                       "yolo_download.py")):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise ImportError(
                "the canonical Ultralytics YOLO sample was not found above "
                f"{os.path.dirname(os.path.abspath(__file__))}; this "
                "compatibility entry point forwards to it and cannot run "
                "without it.")
        current = parent


_SAMPLE_PYTHON = os.path.join(_find_sample_root(), "runtime", "python")
if _SAMPLE_PYTHON not in sys.path:
    sys.path.insert(0, _SAMPLE_PYTHON)

from dataclasses import dataclass, field  # noqa: E402
from typing import List, Optional  # noqa: E402

import importlib.util
_module_name = "_rdk_shared_yolo_pose"
if _module_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_module_name, os.path.join(_SAMPLE_PYTHON, "yolo_pose.py"))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module
    _spec.loader.exec_module(_module)
_BaseConfig = sys.modules[_module_name].YoloPoseConfig
_BaseModel = sys.modules[_module_name].YoloPose
from yolo_platform import PlatformProfile, resolve_platform  # noqa: E402


@dataclass
class UltralyticsYOLOPoseConfig(_BaseConfig):
    """UltralyticsYOLOPoseConfig bound to RDK X5.

    Every field is inherited from `YoloPoseConfig`. Only the defaults the
    RDK X5 tree documented differently are reinstated, so the constructor stays
    interchangeable.
    """

    nms_thres: float = 0.70
    classes_num: int = 1

    platform: Optional[PlatformProfile] = field(
        default_factory=lambda: resolve_platform("x5"))


class UltralyticsYOLOPose(_BaseModel):
    """UltralyticsYOLOPoseConfig counterpart of `YoloPose` bound to RDK X5."""

    def post_process(self, *args, **kwargs):
        import numpy as np
        boxes, scores, _ids, xy, confidence = super().post_process(*args, **kwargs)
        return boxes, scores, np.concatenate([xy, confidence], axis=-1)

__all__ = ["UltralyticsYOLOPose", "UltralyticsYOLOPoseConfig"]
