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

"""Compatibility import path for the readable DFL detector.

The implementation lives in :mod:`...runtime.python.detect`
(``YoloDetect`` with the visible preprocess → infer → postprocess →
predict flow); this module keeps the established ``yolo_detect`` import
surface, including the task dispatch that resolves this module by name.
``pre_process`` / ``forward`` / ``post_process`` remain thin aliases on
the class itself.
"""

from __future__ import annotations

from samples.vision.ultralytics_yolo.runtime.python.detect import (  # noqa: F401 - re-exported surface
    DetectionResult,
    YoloDetect,
    YoloDetectConfig,
)

__all__ = ["DetectionResult", "YoloDetectConfig", "YoloDetect"]
