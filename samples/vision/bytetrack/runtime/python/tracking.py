# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Stateful person tracking around the unified S YOLOv5 detector."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TrackingConfig:
    """Source tracking defaults; frame_rate scales track_buffer/30.

    Thresholds are finite [0,1], buffer is a nonnegative integer, and frame_rate
    is a positive integer. mot20 preserves the source optional matching mode.
    """
    track_thresh: float = .3
    track_buffer: int = 60
    match_thresh: float = .8
    frame_rate: int = 30
    mot20: bool = False

    def __post_init__(self):
        for value in (self.track_thresh,self.match_thresh):
            if isinstance(value,bool) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Tracking thresholds must be finite [0,1].')
        if type(self.track_buffer) is not int or self.track_buffer<0:raise ValueError('track_buffer must be a nonnegative integer.')
        if type(self.frame_rate) is not int or self.frame_rate<=0:raise ValueError('frame_rate must be a positive integer.')
        if type(self.mot20) is not bool:raise ValueError('mot20 must be bool.')


@dataclass(frozen=True)
class Track:
    """Owned snapshot: ID, F64 XYXY tuple in original pixels, score and frame index.

    IDs use the source process-global increasing counter. Track objects returned
    by one update do not change when later frames update internal Kalman state.
    """
    track_id: int
    tlbr: tuple[float,float,float,float]
    score: float
    frame_id: int

    @property
    def tlwh(self):
        """Return original-coordinate XYWH as an immutable tuple."""
        x1,y1,x2,y2=self.tlbr;return x1,y1,x2-x1,y2-y1


def _create_tracker(config):
    # Optional CPU dependencies are imported only when constructing a stream.
    from .tracker_backend.byte_tracker import BYTETracker
    return BYTETracker(config,frame_rate=config.frame_rate)


class ByteTrackTask:
    """One ordered frame stream. Stateless detector stages + one tracking update.

    pre_process and forward do not advance tracking. post_process decodes,
    keeps only COCO person (class 0), then updates the CPU tracker once, including
    on empty detections. predict composes exactly those three operations.
    Not thread safe; do not share a task across independent videos. reset clears
    history/frame index but does not reset the process-global ID counter. After
    a backend update exception, reset before reusing the stream.
    """
    def __init__(self,detector,*,config=None,tracker_factory=None):
        self.detector=detector;self.config=config or TrackingConfig()
        self._tracker_factory=tracker_factory or _create_tracker
        self.tracker=self._tracker_factory(self.config)

    @property
    def frame_index(self):
        return int(self.tracker.frame_id)

    def reset(self):
        """Discard stream history, preserve detector/parameters, keep IDs monotonic."""
        self.tracker=self._tracker_factory(self.config)

    def pre_process(self,image):
        """Prepare one HWC U8 BGR frame and immutable detector geometry context."""
        return self.detector.pre_process(image)

    def forward(self,tensors):
        """Delegate native tensor inference only; no tracking/state transition."""
        return self.detector.forward(tensors)

    def post_process(self,outputs,context):
        """Decode detections, update tracker once, return tuple of owned Track snapshots."""
        detections=self.detector.post_process(outputs,context)
        # Clipping a detection wholly in letterbox padding can leave zero area.
        # Kalman XYAH initialization divides by height; keep these out of state.
        person=(detections.class_ids==0) & (detections.boxes[:,2]>detections.boxes[:,0]) & (detections.boxes[:,3]>detections.boxes[:,1])
        values=np.column_stack((detections.boxes[person],detections.scores[person])).copy() if np.any(person) else np.empty((0,5),dtype=np.float32)
        shape=context.original_size
        try:tracks=self.tracker.update(values,shape,shape)
        except Exception as exc:raise RuntimeError(f'tracker update failed; reset the stream before reuse: {exc}') from exc
        return tuple(Track(int(t.track_id),tuple(float(x) for x in t.tlbr),float(t.score),self.frame_index) for t in tracks)

    def predict(self,image):
        """Process one ordered frame through the three public stages exactly once."""
        prepared=self.pre_process(image)
        raw=self.forward(prepared.tensors)
        return self.post_process(raw,prepared.context)
