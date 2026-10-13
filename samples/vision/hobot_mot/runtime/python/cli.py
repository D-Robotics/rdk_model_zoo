"""CLI helpers for the HobotMot tracker replay: argument parser, JSONL
loading, rendering and result delivery. The orchestration (construct the
tracker, call predict) stays visible in main.py; the native library is only
touched when a run actually starts, so help and validation need no board
build.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from track import DEFAULT_CONFIG, SAMPLE_ROOT, Detection

DEFAULT_INPUT = SAMPLE_ROOT / "test_data" / "detections.jsonl"


@dataclass
class FrameRecord:
    """One parsed JSONL record: a frame of detections."""

    frame_index: int
    timestamp_ms: int
    image_width: int  # 0 -> use CLI defaults
    image_height: int
    detections: list[Detection]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hobot_mot",
        description="Replay a detection JSONL through the HobotMot (IOU 2.0) "
                    "CPU tracker.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"detection JSONL file (default: {DEFAULT_INPUT})")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help=f"IOU_2.0 tracker config JSON (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output", default=None,
                        help="optional JSON result file")
    parser.add_argument("--frame-width", type=int, default=960,
                        help="frame width when a record has no image size "
                             "(default: 960)")
    parser.add_argument("--frame-height", type=int, default=544,
                        help="frame height when a record has no image size "
                             "(default: 544)")
    parser.add_argument("--native-lib", default=None,
                        help="path of libhobot_mot_capi.so (default: search "
                             "runtime/cpp/build/<target>/; env "
                             "HOBOT_MOT_NATIVE_LIB)")
    return parser


def load_frame_records(input_path: str | Path) -> list[FrameRecord]:
    """Load and structurally validate a detection JSONL file.

    Raises:
        ValueError: naming the file and line of the first bad record.
    """
    path = Path(input_path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot open input file: {exc}") from exc
    records: list[FrameRecord] = []
    last_ts = -1
    last_index = -1
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            root = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: not valid JSON: {exc}") from exc
        where = f"{path}:{line_no}"
        if not isinstance(root, dict):
            raise ValueError(f"{where}: each record must be a JSON object")
        for key in ("frame_index", "timestamp_ms", "detections"):
            if key not in root:
                raise ValueError(f"{where}: missing '{key}'")
        frame_index = root["frame_index"]
        timestamp_ms = root["timestamp_ms"]
        if not _is_int(frame_index) or not _is_int(timestamp_ms):
            raise ValueError(
                f"{where}: 'frame_index' and 'timestamp_ms' must be integers")
        if timestamp_ms <= last_ts:
            raise ValueError(f"{where}: timestamp_ms must strictly increase")
        if frame_index <= last_index:
            raise ValueError(f"{where}: frame_index must strictly increase")
        last_ts = timestamp_ms
        last_index = frame_index
        image_width = image_height = 0
        if "image_width" in root or "image_height" in root:
            image_width = root.get("image_width")
            image_height = root.get("image_height")
            if not _is_int(image_width) or not _is_int(image_height) \
                    or image_width <= 0 or image_height <= 0:
                raise ValueError(
                    f"{where}: 'image_width' and 'image_height' must be "
                    f"positive integers when present")
        detections = root["detections"]
        if not isinstance(detections, list):
            raise ValueError(f"{where}: 'detections' must be an array")
        parsed: list[Detection] = []
        for i, node in enumerate(detections):
            if not isinstance(node, dict) or "bbox" not in node \
                    or "score" not in node or "class_id" not in node:
                raise ValueError(
                    f"{where}: detection {i} needs bbox, score and class_id")
            bbox = node["bbox"]
            score = node["score"]
            class_id = node["class_id"]
            if not isinstance(bbox, list) or len(bbox) != 4 \
                    or not all(isinstance(v, (int, float))
                               and not isinstance(v, bool) for v in bbox):
                raise ValueError(
                    f"{where}: detection {i} bbox must be 4 numbers "
                    f"(x1, y1, x2, y2)")
            if not isinstance(score, (int, float)) or isinstance(score, bool) \
                    or not 0.0 <= float(score) <= 1.0:
                raise ValueError(
                    f"{where}: detection {i} score must be within [0, 1]")
            if not _is_int(class_id):
                raise ValueError(
                    f"{where}: detection {i} class_id must be an integer")
            parsed.append(Detection(tuple(float(v) for v in bbox),
                                    float(score), int(class_id)))
        records.append(FrameRecord(int(frame_index), int(timestamp_ms),
                                   int(image_width), int(image_height),
                                   parsed))
    if not records:
        raise ValueError(f"input file has no frame records: {path}")
    return records


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def render_frames(results, records) -> None:
    """Print one line per frame with ids, boxes, states and removed ids."""
    for record, result in zip(records, results):
        parts = [f"frame {record.frame_index} ts {result.timestamp_ms}:"]
        for track in result.tracks:
            box = ",".join(str(v) for v in track.bbox)
            parts.append(f"[cls {track.class_id} id {track.track_id} "
                         f"({box}) score {track.score:.2f} {track.state}]")
        for gone in result.disappeared:
            parts.append(f"[removed cls {gone.class_id} id {gone.track_id}]")
        print(" ".join(parts))


def save_results(output_path: str | Path, records, results) -> None:
    """Write full results as JSON (frames -> tracks and disappeared ids)."""
    frames = []
    for record, result in zip(records, results):
        frames.append({
            "frame_index": record.frame_index,
            "timestamp_ms": result.timestamp_ms,
            "tracks": [{
                "class_id": track.class_id,
                "track_id": track.track_id,
                "bbox": list(track.bbox),
                "score": track.score,
                "state": track.state,
            } for track in result.tracks],
            "disappeared": [{
                "class_id": gone.class_id,
                "track_id": gone.track_id,
            } for gone in result.disappeared],
        })
    Path(output_path).write_text(json.dumps(frames, indent=2) + "\n",
                                 encoding="utf-8")
