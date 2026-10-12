# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Faster R-CNN command surface: pair selection, listing, dry-run, rendering.

The paired-model descriptor (``model/frcnn_mbv3_320.json``) records, per
target, the exact PartA/PartB artifacts this sample was verified with —
filenames, SHA-256 digests and input layouts.  Artifacts are produced by
the conversion guide (nothing is downloaded): selection resolves
``model/<target>/<file>`` and reports missing files instead of guessing.
Everything here is importable without the board SDK; ``main.py`` composes
these helpers with the task stages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from utils.py_utils.platforms import resolve_target

ROOT = Path(__file__).resolve().parents[5]
SAMPLE_DIR = ROOT / "samples" / "vision" / "fasterrcnn"
MODEL_DIR = SAMPLE_DIR / "model"
DESCRIPTOR_PATH = MODEL_DIR / "frcnn_mbv3_320.json"
DEFAULT_IMAGE = SAMPLE_DIR / "test_data" / "bus.jpg"
DEFAULT_LABELS = MODEL_DIR / "coco91_labels.txt"
DEFAULT_RESULT = SAMPLE_DIR / "test_data" / "result.jpg"

SUPPORTED_TARGETS = ("x5", "s100", "s100p", "s600")
SAMPLE = "fasterrcnn"


class BindingError(ValueError):
    """A requested target or pair selection violates the published contract."""


@dataclass(frozen=True)
class TargetPair:
    """One target's descriptor row: file names, layout and hashes."""
    target: str
    march: str
    parta_file: str
    parta_sha256: str
    partb_file: str
    partb_sha256: str
    partb_layout: str
    compiler_latency_us: dict

    @property
    def parta_path(self) -> Path:
        return MODEL_DIR / self.target / self.parta_file

    @property
    def partb_path(self) -> Path:
        return MODEL_DIR / self.target / self.partb_file

    @property
    def asset_id(self) -> str:
        return f"{self.target}:{SAMPLE}:{self.parta_file}+{self.partb_file}"


def load_descriptor() -> dict:
    """Read the paired-model descriptor; malformed files raise BindingError."""
    try:
        data = json.loads(DESCRIPTOR_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise BindingError(f"paired-model descriptor unreadable: {DESCRIPTOR_PATH} ({exc})") from exc
    if data.get("sample") != SAMPLE or "targets" not in data:
        raise BindingError("paired-model descriptor is missing its sample/targets tables.")
    return data


def descriptor_pairs() -> dict[str, TargetPair]:
    """Map every supported target to its descriptor row."""
    data = load_descriptor()
    pairs: dict[str, TargetPair] = {}
    for target, row in data["targets"].items():
        partb = row["partb"]
        pairs[target] = TargetPair(
            target=target,
            march=row["march"],
            parta_file=row["parta"]["file"],
            parta_sha256=row["parta"]["sha256"],
            partb_file=partb["file"],
            partb_sha256=partb["sha256"],
            partb_layout=partb["layout"],
            compiler_latency_us=dict(row.get("compiler_latency_us", {})),
        )
    missing = set(SUPPORTED_TARGETS) - set(pairs)
    if missing:
        raise BindingError(f"descriptor lacks targets: {sorted(missing)}.")
    return pairs


def resolve_selection(target: str = "auto", *, parta_path=None, partb_path=None,
                       soc_name=None, board_type=None):
    """Resolve the PartA/PartB pair for one target.

    Args:
        target: ``auto`` resolves the executing board; x5/s100/s100p/s600
            select explicitly.
        parta_path: Explicit PartA artifact (overrides the descriptor path;
            layout facts still come from the descriptor).
        partb_path: Explicit PartB artifact.

    Returns:
        PairedSelection: Concrete target, artifact paths and PartB layout.

    Raises:
        BindingError: Unknown target, mixed explicit/implicit pair halves,
            or a target the descriptor does not publish.
    """
    from samples.vision.fasterrcnn.runtime.python.detect import (
        PairedSelection,
    )

    resolved = resolve_target(target, soc_name=soc_name, board_type=board_type)
    if resolved not in SUPPORTED_TARGETS:
        raise BindingError(
            f"Faster R-CNN publishes {list(SUPPORTED_TARGETS)}, not {resolved!r}.")
    pair = descriptor_pairs()[resolved]
    if (parta_path is None) != (partb_path is None):
        raise BindingError("--parta-path and --partb-path must be given together.")
    custom = parta_path is not None
    return PairedSelection(
        target=resolved,
        parta_path=Path(parta_path) if parta_path else pair.parta_path,
        partb_path=Path(partb_path) if partb_path else pair.partb_path,
        partb_layout=pair.partb_layout,
        asset_id=f"{resolved}:{SAMPLE}:custom-pair" if custom else pair.asset_id,
        # Descriptor-recorded digests gate artifact verification; a custom
        # pair carries none and must satisfy the live SDK metadata instead.
        parta_sha256=None if custom else pair.parta_sha256,
        partb_sha256=None if custom else pair.partb_sha256,
    )


def list_available_assets(target: str | None = None) -> tuple[TargetPair, ...]:
    """List descriptor pairs; ``target`` filters (auto lists everything)."""
    pairs = descriptor_pairs()
    if target in (None, "auto"):
        return tuple(pairs[t] for t in SUPPORTED_TARGETS)
    if target not in SUPPORTED_TARGETS:
        raise BindingError(f"Faster R-CNN publishes {list(SUPPORTED_TARGETS)}, not {target!r}.")
    return (pairs[target],)


def build_parser() -> argparse.ArgumentParser:
    """Options for the image/video entry point."""
    parser = argparse.ArgumentParser(
        prog="fasterrcnn",
        description="TorchVision Faster R-CNN MobileNetV3-Large-320-FPN on RDK boards")
    parser.add_argument("--target", default="auto",
                        choices=("auto",) + SUPPORTED_TARGETS,
                        help="board target; auto resolves the running board")
    parser.add_argument("--parta-path", default=None, type=Path,
                        help="explicit PartA artifact (.bin on X5, .hbm on S)")
    parser.add_argument("--partb-path", default=None, type=Path,
                        help="explicit PartB artifact; pair with --parta-path")
    parser.add_argument("--input", "-i", default=str(DEFAULT_IMAGE), type=Path,
                        help="input image (or video with --video)")
    parser.add_argument("--video", default=None, type=Path,
                        help="ordered person video; enables tracking")
    parser.add_argument("--track", action="store_true",
                        help="track persons across --video frames")
    parser.add_argument("--reset-every", default=None, type=int, metavar="N",
                        help="reset the tracker every N frames "
                             "(ids stay monotonic across resets)")
    parser.add_argument("--records", default=None, type=Path,
                        help="write per-frame track records as TSV "
                             "(frame, track_id, x1, y1, x2, y2, score)")
    parser.add_argument("--no-encode", action="store_true",
                        help="skip the annotated video writer (timing-only "
                             "runs); the report's result_path is null")
    parser.add_argument("--save-path", default=str(DEFAULT_RESULT), type=Path,
                        help="annotated output image path")
    parser.add_argument("--threads", default=None, type=int, choices=(1, 2, 4),
                        help="CPU thread count for the resize/convert stages")
    parser.add_argument("--label-file", default=str(DEFAULT_LABELS), type=Path,
                        help="91-slot COCO label file (line i names label i)")
    parser.add_argument("--list-models", action="store_true",
                        help="list descriptor pairs and local availability")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved selection without loading models")
    return parser


def list_models(target: str | None = None) -> int:
    """Print descriptor pairs with local file presence; return shell status."""
    rows = []
    for pair in list_available_assets(target):
        rows.append({
            "target": pair.target,
            "march": pair.march,
            "asset_id": pair.asset_id,
            "parta": str(pair.parta_path),
            "parta_present": pair.parta_path.is_file(),
            "partb": str(pair.partb_path),
            "partb_present": pair.partb_path.is_file(),
            "partb_layout": pair.partb_layout,
            "compiler_latency_us": pair.compiler_latency_us,
        })
    print(json.dumps(rows, indent=2))
    return 0


def dry_run(selection, args=None) -> int:
    """Print the resolved pair without importing the board SDK."""
    print(json.dumps({
        "target": selection.target,
        "parta_path": str(selection.parta_path),
        "parta_present": Path(selection.parta_path).is_file(),
        "partb_path": str(selection.partb_path),
        "partb_present": Path(selection.partb_path).is_file(),
        "partb_layout": selection.partb_layout,
        "asset_id": selection.asset_id,
    }, indent=2))
    return 0


def load_labels(path: Path) -> list[str]:
    """Read the 91-slot label file; line ``i`` names label ``i``."""
    text = Path(path).read_text().splitlines()
    if len(text) != 91:
        raise ValueError(f"label file must have 91 lines, {path} has {len(text)}.")
    return text


def render_detections(image, result, labels):
    """Draw xyxy boxes with label names and scores onto a copy of ``image``."""
    import cv2

    canvas = image.copy()
    for box, score, label in zip(result.boxes, result.scores, result.labels):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 215, 255), 2)
        text = f"{labels[label]} {score:.2f}"
        scale, thickness = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 215, 255), -1)
        cv2.putText(canvas, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def render_tracks(image, tracks):
    """Draw tracked person boxes with ids onto a copy of ``image``."""
    import cv2

    canvas = image.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in track.tlbr)
        color = tuple(int(c) for c in ((90 * track.track_id) % 255, 200, 255 - (60 * track.track_id) % 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"id{track.track_id} {track.score:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def save_result(path: Path, canvas) -> None:
    """Write the annotated image, creating parent directories as needed."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"failed to write {path}")


def require_pair_files(selection) -> None:
    """Raise FileNotFoundError naming the conversion guide for a missing half."""
    if not Path(selection.parta_path).is_file():
        raise FileNotFoundError(
            f"PartA artifact not found: {selection.parta_path} "
            "(compile it with the conversion guide, or pass --parta-path)")
    if not Path(selection.partb_path).is_file():
        raise FileNotFoundError(
            f"PartB artifact not found: {selection.partb_path} "
            "(compile it with the conversion guide, or pass --partb-path)")


def verify_pair_hashes(selection) -> dict:
    """Hash both artifacts; enforce the recorded digests for descriptor pairs.

    Returns the observed sha256 digests (and, for descriptor pairs, whether
    each matches) for the run report.  A descriptor-resolved pair must match
    its recorded digest — a mismatch means a stale or foreign artifact under
    a published name, which fails here instead of running unverified bytes.
    A custom ``--parta-path/--partb-path`` pair has no recorded digest; its
    hashes are reported and any recorded fallback in the binding is gated
    on them separately (see :class:`PairedRuntime`).
    """
    from samples.vision.fasterrcnn.runtime.python.detect import file_sha256

    parta_sha = file_sha256(selection.parta_path)
    partb_sha = file_sha256(selection.partb_path)
    result = {
        "parta_sha256": parta_sha,
        "partb_sha256": partb_sha,
        "parta_matches_descriptor": (None if selection.parta_sha256 is None
                                     else parta_sha == selection.parta_sha256),
        "partb_matches_descriptor": (None if selection.partb_sha256 is None
                                     else partb_sha == selection.partb_sha256),
    }
    for half, expected, observed in (("PartA", selection.parta_sha256, parta_sha),
                                     ("PartB", selection.partb_sha256, partb_sha)):
        if expected is not None and observed != expected:
            raise BindingError(
                f"{half} artifact {selection.parta_path if half == 'PartA' else selection.partb_path} "
                f"sha256 {observed} does not match the descriptor's {expected}; "
                "recompile it with the conversion guide or pass an explicit --parta-path/--partb-path pair.")
    return result


def save_detection_image(image, result, label_file: Path, save_path: Path) -> dict:
    """Draw one image result and write it; return the JSON-ready payload."""
    labels = load_labels(Path(label_file).expanduser())
    out_path = Path(save_path).expanduser()
    save_result(out_path, render_detections(image, result, labels))
    return {"boxes": result.boxes.tolist(), "scores": result.scores.tolist(),
            "labels": result.labels.tolist(), "result_path": str(out_path)}


def track_video(tracker, video_path: Path, out_path: Path, *,
                reset_every: int | None = None, records_path: Path | None = None,
                encode: bool = True) -> dict:
    """Run per-frame person tracking over an ordered video.

    Reads ``video_path``, calls ``tracker.predict`` on every frame, draws
    the tracks and writes the annotated mp4 to ``out_path`` (parent
    directories are created, and the writer must open before any frame is
    processed — a requested output that cannot be written fails the run
    instead of silently producing a missing file).  With ``reset_every``
    the tracker resets every N frames (ids stay monotonic); with
    ``records_path`` a per-frame track TSV is written
    (``%.17g`` values, full precision for cross-language comparison).
    ``encode=False`` (``--no-encode``) skips the writer entirely for
    timing-only runs and reports ``result_path: None``.  Returns the frame
    count, output path and the measured timing summary for the caller's
    JSON report.
    """
    import time

    import cv2

    if reset_every is not None and reset_every < 1:
        raise ValueError("--reset-every must be a positive frame count.")
    video_path, out_path = Path(video_path), Path(out_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"video not found or unreadable: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if encode:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise OSError(f"failed to open output video for writing: {out_path}")
    records_file = None
    frames = 0
    tracks_total = 0
    resets = 0
    predict_ms: list[float] = []
    tracker_ms: list[float] = []
    proposals: list[int] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    try:
        if records_path is not None:
            records_file = open(Path(records_path).expanduser(), "w",
                                encoding="utf-8", newline="\n")
            records_file.write("frame\ttrack_id\tx1\ty1\tx2\ty2\tscore\n")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            tracks = tracker.predict(frame)
            predict_ms.append((time.perf_counter() - started) * 1e3)
            tracker_ms.append(tracker.last_tracker_us / 1e3)
            proposals.append(tracker.detector.last_proposals)
            tracks_total += len(tracks)
            if records_file is not None:
                for track in tracks:
                    records_file.write(
                        f"{frames}\t{track.track_id}\t" +
                        "\t".join(format(v, ".17g") for v in (*track.tlbr, track.score))
                        + "\n")
            if writer is not None:
                writer.write(render_tracks(frame, tracks))
            frames += 1
            if reset_every is not None and frames % reset_every == 0:
                tracker.reset()
                resets += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if records_file is not None:
            records_file.close()
    wall_ms = (time.perf_counter() - wall_start) * 1e3
    return {
        "video": str(video_path),
        "frames": frames,
        "predict_ms_mean": sum(predict_ms) / max(len(predict_ms), 1),
        "predict_ms_min": min(predict_ms) if predict_ms else 0.0,
        "predict_ms_max": max(predict_ms) if predict_ms else 0.0,
        "tracker_ms_mean": sum(tracker_ms) / max(len(tracker_ms), 1),
        "cpu_ms_total": (time.process_time() - cpu_start) * 1e3,
        "wall_ms_total": wall_ms,
        "proposals_mean": sum(proposals) / max(len(proposals), 1),
        "tracks_total": tracks_total,
        "reset_every": reset_every,
        "resets": resets,
        "records_path": str(records_path) if records_path is not None else None,
        "result_path": str(out_path) if writer is not None else None,
    }


__all__ = [
    "BindingError", "DEFAULT_IMAGE", "DEFAULT_LABELS", "DEFAULT_RESULT",
    "MODEL_DIR", "SAMPLE_DIR", "SUPPORTED_TARGETS", "TargetPair",
    "build_parser", "descriptor_pairs", "dry_run", "list_available_assets",
    "list_models", "load_descriptor", "load_labels", "render_detections",
    "render_tracks", "require_pair_files", "resolve_selection",
    "save_detection_image", "save_result", "track_video", "verify_pair_hashes",
]
