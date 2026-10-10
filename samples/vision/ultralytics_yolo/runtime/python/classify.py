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
"""YOLO image classification: stages and Softmax/Top-K decode.

:class:`YoloCls` owns the readable classification stages (preprocess →
infer → postprocess) with the validated floating logit transport; below the
class lives the numeric Softmax/Top-K postprocess.  The shared image
transport comes from ``detect.py``; the classification contract and runner
come from ``backend.py``.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np
from scipy.special import softmax

from samples.vision.ultralytics_yolo.runtime.python.backend import (
    ClassificationContract,
    ModelSelection,
    build_runner,
)
from samples.vision.ultralytics_yolo.runtime.python.cli import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    _forward_runner,
    _prepare_image,
    _set_scheduling_params,
    _size_from_runner,
)

# ====================================================================
# The classification task class.
# ====================================================================

@dataclass
class YoloClsConfig:
    """Local .bin/.hbm path, target, resize (0 stretch/1 letterbox), and Top-K.

    Library default is stretch on every target; the CLI explicitly applies its
    family/platform resize policy. Published models have 1000 ImageNet classes.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    topk: int = 5
    resize_type: int = 0
    contract: Optional[ClassificationContract] = None


class YoloCls:
    """A single-image classifier with validated floating raw output transport."""

    def __init__(self, config: YoloClsConfig, runner=None):
        self.cfg = config
        requested = config.contract or ClassificationContract()
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="classify",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        if self.binding is None or self.binding.contract.task != "classify":
            raise ValueError("YoloCls requires a runner with a classification binding.")
        self.contract = self.binding.contract
        self.model = getattr(runner, "model", runner)
        self.model_name = runner.model_name
        self.input_adapter = runner.input_adapter
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        self.input_names = tuple(runner.input_names)
        self.output_names = tuple(runner.output_names)
        self.input_shapes = dict(runner.input_shapes)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        """Delegate explicit scheduling values to the runtime boundary."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores
        )

    def preprocess(self, img, image_format="BGR"):
        """Validate BGR uint8 HxWx3 and return the bound nested NV12 input map."""
        tensors, _ = _prepare_image(
            self.runner,
            self.input_adapter,
            self.input_size,
            self.cfg.resize_type,
            img,
            image_format,
        )
        return tensors

    def infer(self, input_tensor):
        """Execute once; return the borrowed physical floating logits unchanged."""
        return _forward_runner(self.runner, input_tensor)

    def postprocess(self, outputs, topk=None):
        """Validate the raw output and return independent (class ID, probability) pairs."""
        logits = self.binding.read_raw_outputs(outputs)["logits"]
        return classification_topk(logits, self.cfg.topk if topk is None else topk)

    def predict(self, img, image_format="BGR", topk=None):
        """Compose the three public stages for one image."""
        prepared = self.preprocess(img, image_format)
        outputs = self.infer(prepared)
        return self.postprocess(outputs, topk)

    def __call__(self, img, image_format="BGR", topk=None):
        return self.predict(img, image_format, topk)

    # ------------------------------------------------------------------
    # Compatibility surface: the established stage names stay thin
    # aliases of the implementations above (no second implementation).
    # ------------------------------------------------------------------

    def pre_process(self, img, image_format="BGR"):
        """Compatibility alias for :meth:`preprocess`."""
        return self.preprocess(img, image_format)

    def forward(self, input_tensor):
        """Compatibility alias for :meth:`infer`."""
        return self.infer(input_tensor)

    def post_process(self, outputs, topk=None):
        """Compatibility alias for :meth:`postprocess`."""
        return self.postprocess(outputs, topk)

# ====================================================================
# Numeric decode: classification Softmax/Top-K; no SDK, labels or I/O.
# ====================================================================

def classification_topk(logits, topk):
    """Apply the source Softmax/sort rule; clamp positive K to the class count.

    Exact ties follow NumPy's existing argsort order, not a new class-ID policy.
    Scalar Python results cannot alias a reusable SDK output buffer.
    """
    if (
        isinstance(topk, (bool, np.bool_))
        or not isinstance(topk, Integral)
        or topk <= 0
    ):
        raise ValueError("topk must be a positive integer.")
    values = np.asarray(logits)
    if (
        not np.issubdtype(values.dtype, np.floating)
        or not values.size
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Classification logits must be nonempty finite floating values."
        )
    probabilities = softmax(values.reshape(-1))
    indices = np.argsort(probabilities)[::-1][:topk]
    return [(int(index), float(probabilities[index])) for index in indices]

__all__ = ["YoloCls", "YoloClsConfig", "classification_topk"]
