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

"""Classification stages: image preparation, raw inference, Softmax/Top-K."""

from dataclasses import dataclass
from typing import Optional, Tuple
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    ClassificationContract,
    ModelSelection,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    _prepare_image,
    _forward_runner,
    _set_scheduling_params,
    _size_from_runner,
)
from samples.vision.ultralytics_yolo.runtime.python.classification_decode import (
    classification_topk,
)


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
