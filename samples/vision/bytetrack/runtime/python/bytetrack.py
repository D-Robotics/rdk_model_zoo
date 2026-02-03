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

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from yolov5 import YoloV5X, YOLOv5Config
from tracker.byte_tracker import BYTETracker

@dataclass
class ByteTrackConfig(YOLOv5Config):
    """Configuration for ByteTrack, extending YOLOv5Config.

    This dataclass includes both detection parameters (inherited from YOLOv5Config)
    and specific tracking parameters used by BYTETracker.

    Attributes:
        track_thresh: Confidence threshold for tracking. Detections with confidence
            above this value are considered 'high score' detections. Default: 0.3.
        track_buffer: Number of frames to keep a lost track alive. Default: 60.
        match_thresh: IoU matching threshold. Default: 0.8.
        frame_rate: Frame rate of the input video. Default: 30.
        mot20: Whether to use MOT20 settings (experimental). Default: False.
    """
    # Tracking parameters
    track_thresh: float = 0.3
    track_buffer: int = 60
    match_thresh: float = 0.8
    frame_rate: int = 30
    mot20: bool = False

class ByteTrackArgs:
    """Helper class to pass arguments to BYTETracker.

    Wraps the configuration parameters into an object structure expected
    by the BYTETracker implementation.

    Args:
        config: ByteTrackConfig object containing tracking parameters.
    """
    def __init__(self, config: ByteTrackConfig):
        self.track_thresh = config.track_thresh
        self.track_buffer = config.track_buffer
        self.match_thresh = config.match_thresh
        self.mot20 = config.mot20
        self.frame_rate = config.frame_rate

class ByteTrack:
    """ByteTrack wrapper containing a YOLO detector and BYTETracker.

    This class orchestrates the Multi-Object Tracking (MOT) pipeline:
    1. Uses `YoloV5X` to detect objects in the current frame.
    2. Filters detections to a specific class (Person).
    3. Updates the `BYTETracker` with the filtered detections.
    """

    def __init__(self, config: ByteTrackConfig):
        """Initialize the ByteTrack pipeline.

        Args:
            config: Configuration object containing both detection and tracking parameters.
        """
        self.cfg = config
        
        # 1. Initialize Detector (YOLOv5)
        self.detector = YoloV5X(config)
        
        # 2. Initialize Tracker
        tracker_args = ByteTrackArgs(config)
        self.tracker = BYTETracker(tracker_args)

    def set_scheduling_params(self, priority: Optional[int] = None, bpu_cores: Optional[list] = None):
        """Configure inference scheduling parameters for the underlying detector.

        Args:
            priority: Inference priority in the range [0, 255].
            bpu_cores: List of BPU core indices used for inference.
        """
        self.detector.set_scheduling_params(priority, bpu_cores)

    def predict(self, img: np.ndarray) -> List:
        """Run detection and tracking on a single image.

        Args:
            img: Input image array (BGR format).

        Returns:
            A list of tracked targets (STrack objects). Each target typically contains
            fields like `tlbr` (top-left-bottom-right box coordinates) and `track_id`.
        """
        ori_h, ori_w = img.shape[:2]
        
        # 1. Detection (using YOLOv5)
        # xyxy: (N, 4), score: (N,), cls: (N,)
        xyxy, scores, cls_ids = self.detector.predict(img)
        
        # 2. Filter for Person class (Class ID 0)
        # Note: YOLOv5 COCO Class 0 is Person
        person_mask = (cls_ids == 0)
        
        detections = []
        if np.any(person_mask):
            p_xyxy = xyxy[person_mask]
            p_scores = scores[person_mask]
            
            # Combine into (N, 5) array: [x1, y1, x2, y2, score]
            # p_scores needs to be reshaped to (N, 1)
            detections = np.hstack((p_xyxy, p_scores[:, np.newaxis]))
        else:
            detections = np.empty((0, 5))
            
        # 3. Update Tracker
        # BYTETracker expects detections as (N, 5) array
        # Note: self.detector.predict returns coordinates in original image scale.
        # We pass (ori_h, ori_w) as img_size to tracker.update so that the internal scale factor becomes 1.0.
        online_targets = self.tracker.update(detections, (ori_h, ori_w), (ori_h, ori_w))
            
        return online_targets
