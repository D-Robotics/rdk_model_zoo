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
    """
    @brief Configuration for ByteTrack, extending YOLOv5Config.
    """
    # Tracking parameters
    track_thresh: float = 0.3
    track_buffer: int = 60
    match_thresh: float = 0.8
    frame_rate: int = 30
    mot20: bool = False

class ByteTrackArgs:
    """Helper class to pass arguments to BYTETracker"""
    def __init__(self, config: ByteTrackConfig):
        self.track_thresh = config.track_thresh
        self.track_buffer = config.track_buffer
        self.match_thresh = config.match_thresh
        self.mot20 = config.mot20
        self.frame_rate = config.frame_rate

class ByteTrack:
    """
    @brief ByteTrack wrapper containing a YOLO detector and BYTETracker.
    """
    def __init__(self, config: ByteTrackConfig):
        self.cfg = config
        
        # 1. Initialize Detector (YOLOv5)
        self.detector = YoloV5X(config)
        
        # 2. Initialize Tracker
        tracker_args = ByteTrackArgs(config)
        self.tracker = BYTETracker(tracker_args)

    def set_scheduling_params(self, priority: Optional[int] = None, bpu_cores: Optional[list] = None):
        self.detector.set_scheduling_params(priority, bpu_cores)

    def predict(self, img: np.ndarray) -> List:
        """
        @brief Run detection and tracking on a single image.
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