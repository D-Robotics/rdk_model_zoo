# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Person tracking around the Faster R-CNN detector.

The tracker core is the shared CPU ByteTrack implementation from the
ByteTrack sample (``tracker_backend``); nothing there is modified.  The only
adaptation lives here, at the detector boundary: this sample keeps original
COCO category ids, so *person* is label 1 and is filtered explicitly before
the tracker — the shared ``ByteTrackTask`` wrapper assumes the YOLO
convention (person = class 0) and is intentionally not reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from samples.vision.fasterrcnn.runtime.python.detect import (
    PERSON_LABEL,
    FasterRCNNTask,
)


@dataclass(frozen=True)
class TrackingConfig:
    """ByteTrack source defaults; ``frame_rate`` scales the track buffer."""
    track_thresh: float = 0.3
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30
    mot20: bool = False


@dataclass(frozen=True)
class Track:
    """One tracked person: id, XYXY box in original pixels, score, frame."""
    track_id: int
    tlbr: tuple[float, float, float, float]
    score: float
    frame_id: int

    @property
    def tlwh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.tlbr
        return x1, y1, x2 - x1, y2 - y1


def _create_tracker(config: TrackingConfig) -> Any:
    from samples.vision.bytetrack.runtime.python.tracker_backend.byte_tracker import (
        BYTETracker,
    )
    return BYTETracker(config, frame_rate=config.frame_rate)


class PersonTrackerTask:
    """One ordered frame stream: detect, keep label-1 rows, update tracker.

    ``preprocess``/``infer``/``postprocess`` expose the detector stages
    without touching tracker state; ``postprocess`` decodes detections,
    drops non-person labels and zero-area boxes, then advances the tracker
    exactly once (including on empty frames).  ``predict`` composes the
    three stages.  Not thread safe; ``reset`` clears history but keeps the
    process-global id counter monotonic, matching the shared core.
    """

    def __init__(self, detector: FasterRCNNTask, *, config: TrackingConfig | None = None,
                 tracker_factory=None):
        self.detector = detector
        self.config = config or TrackingConfig()
        self._tracker_factory = tracker_factory or _create_tracker
        self.tracker = self._tracker_factory(self.config)
        self.last_tracker_us: float = 0.0

    @property
    def frame_index(self) -> int:
        return int(self.tracker.frame_id)

    def reset(self) -> None:
        """Discard stream history; detector, parameters and id monotonicity stay."""
        self.tracker = self._tracker_factory(self.config)

    def preprocess(self, image):
        return self.detector.preprocess(image)

    def infer(self, prepared):
        return self.detector.infer(prepared)

    def postprocess(self, outputs, context):
        import numpy as np

        result = self.detector.postprocess(outputs, context)
        person = (result.labels == PERSON_LABEL) & \
                 (result.boxes[:, 2] > result.boxes[:, 0]) & \
                 (result.boxes[:, 3] > result.boxes[:, 1])
        if np.any(person):
            values = np.column_stack((result.boxes[person], result.scores[person])).copy()
        else:
            values = np.empty((0, 5), dtype=np.float32)
        shape = context.original_size
        import time

        started = time.perf_counter()
        try:
            tracks = self.tracker.update(values, shape, shape)
        except Exception as exc:
            raise RuntimeError(
                f"tracker update failed; reset the stream before reuse: {exc}") from exc
        finally:
            self.last_tracker_us = (time.perf_counter() - started) * 1e6
        return tuple(Track(int(t.track_id), tuple(float(v) for v in t.tlbr),
                           float(t.score), self.frame_index) for t in tracks)

    def predict(self, image):
        prepared = self.preprocess(image)
        raw = self.infer(prepared)
        return self.postprocess(raw, prepared.context)


__all__ = ["PersonTrackerTask", "Track", "TrackingConfig"]
