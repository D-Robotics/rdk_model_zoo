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

"""YOLOv8/11 pose stages with raw inference and explicit per-image geometry."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLPoseContract,
    ModelSelection,
)
from samples.vision.ultralytics_yolo.runtime.python.model_runner import build_runner
from samples.vision.ultralytics_yolo.runtime.python.detection_io import (
    PreparedDetection,
    _prepare_image,
    _forward_runner,
    _semantic_outputs,
    _transform_for_postprocess,
    _set_scheduling_params,
    _size_from_runner,
    _normalise_grids,
    _predict_task,
)
from samples.vision.ultralytics_yolo.runtime.python.legacy import (
    pre_process_with_transform,
)
from samples.vision.ultralytics_yolo.runtime.python.pose_decode import decode_pose


@dataclass
class YoloPoseConfig:
    """Configuration for initializing the YoloPose model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO pose
    estimation pipeline. It applies to DFL-based YOLO pose models (v8 and v11).

    Attributes:
        model_path: Path to the compiled YOLO-Pose `.hbm` model.
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        nkpt: Number of keypoints the model predicts. Defaults to 17 (COCO).
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = None
    reg: int = 16
    nkpt: int = 17
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None

    contract: Optional[DFLPoseContract] = None


class YoloPose:
    """DFL COCO pose with owned five-tuple results and probabilities by default."""

    def __init__(self, config: YoloPoseConfig, runner=None):
        self.cfg = config
        requested = config.contract or DFLPoseContract(
            reg_bins=config.reg, strides=config.strides, nkpt=config.nkpt
        )
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="pose",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        self.contract = getattr(self.binding, "contract", requested)
        if self.contract.task != "pose":
            raise ValueError("YoloPose requires a pose contract.")
        self.model = getattr(runner, "model", runner)
        self.model_name = getattr(runner, "model_name", None)
        self.input_adapter = getattr(runner, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        grids = _normalise_grids(
            config.anchor_sizes, self.input_size, self.contract.strides
        )
        if any(h != w for h, w in grids):
            raise ValueError("Published DFL pose requires square input grids.")
        self.anchor_sizes = [h for h, w in grids]
        self.nkpt = self.contract.nkpt
        self.input_names = tuple(getattr(runner, "input_names", ()))
        self.output_names = tuple(getattr(runner, "output_names", ()))
        self.input_shapes = dict(getattr(runner, "input_shapes", {}))
        if config.nms_thres is None:
            config.nms_thres = getattr(config.platform, "nms_thres", 0.7)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        """Forward explicitly supplied scheduling values through the runner."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores
        )

    pre_process_with_transform = pre_process_with_transform

    def pre_process(self, img, image_format="BGR") -> PreparedDetection:
        """Validate BGR uint8 HxWx3 and prepare NV12 with immutable geometry."""
        tensors, transform = _prepare_image(
            self.runner,
            self.input_adapter,
            self.input_size,
            self.cfg.resize_type,
            img,
            image_format,
        )
        return PreparedDetection(tensors, transform)

    def forward(self, input_tensor):
        """Execute the runner once without changing raw output values or layout."""
        return _forward_runner(self.runner, input_tensor)

    def post_process(
        self,
        outputs,
        ori_img_w=None,
        ori_img_h=None,
        score_thres=None,
        nms_thres=None,
        transform=None,
    ):
        """Decode matching image geometry and return boxes/scores/IDs/xy/visibility.

        Visibility is sigmoid probability in the maintained Ultralytics protocol.
        """
        context = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h, self.input_size, self.cfg.resize_type
        )
        semantic = _semantic_outputs(self.binding, self.contract, outputs, "DFL pose")
        return decode_pose(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres if score_thres is None else score_thres,
            self.cfg.nms_thres if nms_thres is None else nms_thres,
        )

    def predict(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        """Compose all three stages; keep the same explicit visibility domain."""
        return _predict_task(self, img, image_format, score_thres, nms_thres)

    def __call__(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        return self.predict(img, image_format, score_thres, nms_thres)
