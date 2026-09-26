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

"""YOLOv8/9/11 instance segmentation: explicit preprocessing, raw call, decoding."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.yolo_platform import PlatformProfile
from samples.vision.ultralytics_yolo.runtime.python.model_binding import (
    DFLSegmentationContract,
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
from samples.vision.ultralytics_yolo.runtime.python.segmentation_decode import (
    decode_segmentation,
)


@dataclass
class YoloSegConfig:
    """Configuration for initializing the YoloSeg model.

    This dataclass stores the model path and all runtime parameters required
    for preprocessing, inference, and postprocessing in the YOLO segmentation
    pipeline. It applies to DFL-based YOLO segmentation models (v8, v9 and v11).

    Attributes:
        model_path: Path to the compiled YOLO-Seg `.hbm` model.
        classes_num: Number of detection classes. Defaults to 80 (COCO).
        resize_type: Image resize mode used during preprocessing.
            - 1: Keep aspect ratio with letterbox padding.
        score_thres: Minimum confidence threshold for filtering detections.
        nms_thres: IoU threshold used for Non-Maximum Suppression.
        reg: Number of DFL regression bins per bounding-box side. Defaults to 16.
        mces_num: Dimension of the MCES (mask coefficient) vector. Defaults to 32.
        strides: Feature map downsampling strides for each detection scale.
        anchor_sizes: Feature map grid sizes for each detection scale.
        do_morph: Whether to apply morphological opening to clean mask edges.
    """

    model_path: str
    platform: Optional[PlatformProfile] = None
    input_shape: Optional[Tuple[int, int]] = None
    classes_num: int = 80
    resize_type: int = 1
    score_thres: float = 0.25
    nms_thres: Optional[float] = None
    reg: int = 16
    mces_num: int = 32
    strides: list = field(default_factory=lambda: [8, 16, 32])
    anchor_sizes: Optional[list] = None
    do_morph: bool = True

    contract: Optional[DFLSegmentationContract] = None


class YoloSeg:
    """DFL segmentation task with a replaceable raw model runner.

    Loading/metadata and scheduling belong to the runner. Image geometry belongs
    to each PreparedDetection, never a cached last image. Physical output order
    is not a model contract. See runtime/python/README for tuple mask semantics.
    """

    def __init__(self, config: YoloSegConfig, runner=None):
        self.cfg = config
        requested = config.contract or DFLSegmentationContract(
            classes=config.classes_num,
            reg_bins=config.reg,
            strides=config.strides,
            mces_num=config.mces_num,
        )
        if runner is None:
            runner = build_runner(
                ModelSelection(
                    config.model_path,
                    target=getattr(config.platform, "key", None),
                    platform=config.platform,
                    task="segment",
                    contract=requested,
                    input_shape=config.input_shape,
                )
            )
        self.runner = runner
        self.binding = getattr(runner, "binding", None)
        self.contract = getattr(self.binding, "contract", requested)
        if self.contract.task != "segment":
            raise ValueError("YoloSeg requires a segmentation contract.")
        self.model = getattr(runner, "model", runner)
        self.model_name = getattr(runner, "model_name", None)
        self.input_adapter = getattr(runner, "input_adapter", None)
        self.input_h, self.input_w = _size_from_runner(runner, config)
        self.input_size = (self.input_h, self.input_w)
        grids = _normalise_grids(
            config.anchor_sizes, self.input_size, self.contract.strides
        )
        if any(h != w for h, w in grids):
            raise ValueError("Published DFL segmentation requires square input grids.")
        self.anchor_sizes = [h for h, w in grids]
        self.input_names = tuple(getattr(runner, "input_names", ()))
        self.output_names = tuple(getattr(runner, "output_names", ()))
        self.input_shapes = dict(getattr(runner, "input_shapes", {}))
        if config.nms_thres is None:
            config.nms_thres = getattr(config.platform, "nms_thres", 0.7)

    def set_scheduling_params(self, priority=None, bpu_cores=None):
        """Forward only explicitly supplied scheduler settings."""
        _set_scheduling_params(
            self.runner, self.model, self.model_name, priority, bpu_cores
        )

    pre_process_with_transform = pre_process_with_transform

    def pre_process(self, img, image_format="BGR") -> PreparedDetection:
        """Return owned NV12 tensors and immutable geometry for BGR uint8 HxWx3."""
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
        """Call the runner once, preserving raw dtype, layout and quantization."""
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
        """Dequantize, decode/NMS and return original-image boxes and ROI masks.

        Pass the matching prepared.transform (preferred) or both original image
        dimensions (legacy). Results own their storage and survive SDK reuse.
        """
        context = _transform_for_postprocess(
            transform, ori_img_w, ori_img_h, self.input_size, self.cfg.resize_type
        )
        semantic = _semantic_outputs(
            self.binding, self.contract, outputs, "DFL segmentation"
        )
        return decode_segmentation(
            semantic,
            self.contract,
            context,
            self.cfg.score_thres if score_thres is None else score_thres,
            self.cfg.nms_thres if nms_thres is None else nms_thres,
            do_morph=self.cfg.do_morph,
        )

    def predict(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        """Compose the same three public stages without I/O or last-image state."""
        return _predict_task(self, img, image_format, score_thres, nms_thres)

    def __call__(self, img, image_format="BGR", score_thres=None, nms_thres=None):
        return self.predict(img, image_format, score_thres, nms_thres)
