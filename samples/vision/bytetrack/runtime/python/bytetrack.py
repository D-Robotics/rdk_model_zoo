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

"""ByteTrack multi-object tracking runtime wrapper for RDK S100/S100P.

This module implements the ByteTrack MOT pipeline on BPU, combining a
YOLOv5x detector with the BYTETracker association algorithm to track
pedestrians across video frames.

Key Features:
    - YOLOv5x BPU detection + BYTETracker association in a single pipeline.
    - Exposes the standard Config + Model interface for consistency with
      other Model Zoo samples.
    - ``pre_process`` / ``forward`` / ``post_process`` delegate to the
      underlying YoloV5X detector; ``ByteTrack.predict`` additionally runs
      the stateful tracker update.

Typical Usage:
    >>> import cv2
    >>> from bytetrack import ByteTrackConfig, ByteTrack
    >>> config = ByteTrackConfig(model_path="../../model/s100/yolov5x_672x672_nv12.hbm")
    >>> tracker = ByteTrack(config)
    >>> frame = cv2.imread("frame.jpg")
    >>> online_targets = tracker.predict(frame)
    >>> for t in online_targets:
    ...     print(t.track_id, t.tlbr)

Notes:
    - BYTETracker is stateful: call ``predict`` once per frame in order.
    - Third-party tracker code lives in ``3rdparty/tracker/``; run.sh adds
      it to PYTHONPATH automatically.
    - Only the Person class (COCO class 0) is tracked by default.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from yolov5 import YoloV5X, YOLOv5Config
from tracker.byte_tracker import BYTETracker


@dataclass
class ByteTrackConfig(YOLOv5Config):
    """Configuration for ByteTrack, extending YOLOv5Config.

    Inherits all detection parameters from YOLOv5Config and adds
    tracking-specific parameters for BYTETracker.

    Attributes:
        track_thresh: Confidence threshold for high-score detections. Default: 0.3.
        track_buffer: Frames to keep a lost track alive. Default: 60.
        match_thresh: IoU threshold for track-detection matching. Default: 0.8.
        frame_rate: Input video frame rate. Default: 30.
        mot20: Enable MOT20 settings (experimental). Default: False.
    """
    track_thresh: float = 0.3
    track_buffer: int = 60
    match_thresh: float = 0.8
    frame_rate: int = 30
    mot20: bool = False


class ByteTrackArgs:
    """Adapter class that converts ByteTrackConfig to the BYTETracker argument format.

    Args:
        config: ByteTrackConfig containing tracking parameters.
    """

    def __init__(self, config: ByteTrackConfig):
        self.track_thresh = config.track_thresh
        self.track_buffer = config.track_buffer
        self.match_thresh = config.match_thresh
        self.mot20 = config.mot20
        self.frame_rate = config.frame_rate


class ByteTrack:
    """ByteTrack multi-object tracking wrapper.

    Combines a YOLOv5x BPU detector with the BYTETracker association
    algorithm. The standard pre_process / forward / post_process methods
    delegate to the underlying YoloV5X detector. predict() additionally
    runs the stateful tracker update.

    Args:
        config: ByteTrackConfig with detection and tracking parameters.

    Notes:
        This wrapper is stateful — the tracker maintains track history
        across frames. Instantiate a new ByteTrack object to reset state.
    """

    def __init__(self, config: ByteTrackConfig):
        """Initialize the YOLOv5 detector and BYTETracker.

        Args:
            config: Configuration for both detection and tracking.
        """
        self.cfg = config
        self.detector = YoloV5X(config)
        tracker_args = ByteTrackArgs(config)
        self.tracker = BYTETracker(tracker_args)

    def set_scheduling_params(
        self,
        priority: Optional[int] = None,
        bpu_cores: Optional[List[int]] = None,
    ) -> None:
        """Set BPU scheduling parameters on the underlying detector.

        Args:
            priority: Inference priority in range [0, 255].
            bpu_cores: List of BPU core indices to use.
        """
        self.detector.set_scheduling_params(priority, bpu_cores)

    def pre_process(
        self, img: np.ndarray, image_format: str = "BGR"
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess one image frame for BPU detection.

        Delegates directly to the underlying YoloV5X detector.

        Args:
            img: Input image as a NumPy array (H, W, 3).
            image_format: Color format of the input image. Default: ``BGR``.

        Returns:
            Dict: Nested input tensor dict accepted by ``forward()``.
        """
        return self.detector.pre_process(img, image_format)

    def forward(
        self, input_tensor: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Run BPU detection inference.

        Delegates directly to the underlying YoloV5X detector.

        Args:
            input_tensor: Prepared input dict from ``pre_process()``.

        Returns:
            Dict: Raw output tensors from hbm_runtime.
        """
        return self.detector.forward(input_tensor)

    def post_process(
        self, outputs: Dict[str, np.ndarray], img: np.ndarray
    ) -> np.ndarray:
        """Decode raw detection outputs to bounding boxes.

        Delegates to the underlying YoloV5X detector's post_process.

        Args:
            outputs: Raw output tensors from ``forward()``.
            img: Original image used to recover input dimensions.

        Returns:
            Tuple: ``(boxes, scores, cls_ids)`` as in the detection spec.
        """
        return self.detector.post_process(outputs, img)

    def predict(self, img: np.ndarray) -> List:
        """Run detection and tracker update for one video frame.

        Orchestrates: pre_process → forward → post_process (person filter)
        → BYTETracker.update.

        Args:
            img: Input video frame (BGR, H x W x 3).

        Returns:
            List[STrack]: Active tracks. Each STrack has ``track_id`` (int)
            and ``tlbr`` (np.ndarray [x1, y1, x2, y2]).
        """
        ori_h, ori_w = img.shape[:2]

        xyxy, scores, cls_ids = self.detector.predict(img)

        # Filter to Person class (COCO class 0)
        person_mask = cls_ids == 0
        if np.any(person_mask):
            detections = np.hstack(
                (xyxy[person_mask], scores[person_mask][:, np.newaxis])
            )
        else:
            detections = np.empty((0, 5))

        # Tracker expects detections in original image scale
        return self.tracker.update(detections, (ori_h, ori_w), (ori_h, ori_w))

    def __call__(self, img: np.ndarray) -> List:
        """Callable alias for ``predict()``.

        Args:
            img: Input video frame (BGR).

        Returns:
            List[STrack]: Same as ``predict()``.
        """
        return self.predict(img)
