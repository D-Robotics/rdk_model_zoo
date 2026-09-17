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
"""RDK S Ultralytics YOLO YOLOv10 detection runtime wrapper (compatibility names).

The maintained implementation lives in the canonical sample at
`samples/vision/ultralytics_yolo/runtime/python/yolo_v10detect.py`. This module keeps
the RDK S module, class and configuration names importable, so existing RDK S
code and documentation keep working, without a second copy of the pipeline.

# The S wrapper documented its own thresholds, anchor sizes and resize
# policy, and had no `platform` field; those defaults are reinstated here.
The runtime behaviour is the canonical behaviour; only the names and the
platform-bound defaults below are reinstated here.

Typical Usage:
    >>> from yolo_v10detect import YoloV10DetectConfig, YoloV10Detect
    >>> net = YoloV10Detect(YoloV10DetectConfig(model_path="yolo11n_detect_nashe_640x640_nv12.hbm"))
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
_REPOSITORY_ROOT = os.path.abspath(os.path.join(_SAMPLE_PYTHON, "../../../../.."))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)
if _SAMPLE_PYTHON not in sys.path:
    sys.path.insert(0, _SAMPLE_PYTHON)

from dataclasses import dataclass, field  # noqa: E402
from typing import List, Optional  # noqa: E402

import importlib.util
_module_name = "_rdk_shared_yolo_v10detect"
if _module_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_module_name, os.path.join(_SAMPLE_PYTHON, "yolo_v10detect.py"))
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module
    _spec.loader.exec_module(_module)
_BaseConfig = sys.modules[_module_name].YoloV10DetectConfig
_BaseModel = sys.modules[_module_name].YoloV10Detect
from yolo_platform import PlatformProfile, resolve_platform  # noqa: E402


@dataclass
class YoloV10DetectConfig(_BaseConfig):
    """YoloV10DetectConfig bound to the RDK S series.

    Every field is inherited from `YoloV10DetectConfig`. Only the defaults the
    RDK S tree documented differently are reinstated, so the constructor stays
    interchangeable.
    """

    resize_type: int = 1
    score_thres: float = 0.25
    anchor_sizes: List[int] = field(
        default_factory=lambda: [80, 40, 20])

    platform: Optional[PlatformProfile] = field(default_factory=resolve_platform)


class YoloV10Detect(_BaseModel):
    """YoloV10DetectConfig counterpart of `YoloV10Detect` bound to the RDK S series."""

    def __init__(self, config: YoloV10DetectConfig):
        """Bind the board platform when the configuration does not name one.

        The RDK S series publishes different artifacts for S100, S100P and
        S600, so a configuration that does not name a platform is bound to the
        board it is constructed on. An unsupported or undetected board fails
        here rather than later, at model load.

        Args:
            config: Runtime configuration, updated in place.
        """
        if config.platform is None:
            config.platform = resolve_platform()
        super().__init__(config)


__all__ = ["YoloV10Detect", "YoloV10DetectConfig"]
