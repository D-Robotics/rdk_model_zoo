# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Frame-level output quality for LiteAvatar renders.

Compares one render against a reference on a per-frame basis: either two
directories of frames (``.png``/``.jpg``, matched by sorted order), or a
frame directory against a reference video.  Frames stream from both sides
in lockstep — only the per-frame metrics are kept, never both full
sequences — and the default full comparison **fails when the frame counts
differ** (a truncated delivery cannot pass).  ``--limit N`` opts into a
clearly labeled prefix check of the first N frames of each side.

Reports per-frame and mean PSNR/MAE (uint8 BGR scale 0-255) and the count
of non-finite or geometry-mismatched frames.  Bit-identical frames have no
finite PSNR; they are counted as ``perfect`` and their ``psnr_db`` is
``null`` — never a capped value — and mean/min PSNR aggregate the frames
with a real MSE (``null`` when every compared frame is perfect).  A
``--gate`` on the mean PSNR turns the run into an acceptance check; a
fully perfect comparison passes any finite gate.  Exit codes: 0 pass,
1 comparison/gate failure, 2 usage/input error.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

FRAME_PATTERN = re.compile(r".*\.(png|jpg|jpeg)$", re.IGNORECASE)


def frame_paths(directory: Path) -> list[Path]:
    """Sorted frame files of a directory (pattern filtered, paired by name)."""

    if not directory.is_dir():
        raise ValueError(f"Not a frame directory: {directory}")
    paths = [item for item in sorted(directory.iterdir())
             if FRAME_PATTERN.match(item.name)]
    if not paths:
        raise ValueError(f"No .png/.jpg frames under {directory}")
    return paths


def stream_frames(source: Path):
    """Yield ``(path_or_label, frame)`` pairs from a directory or a video.

    Directory frames decode one at a time; a decode failure raises with the
    exact offending file name.  Video frames carry an ``f"{source}:{index}"``
    label and decode through an open :class:`cv2.VideoCapture` released in
    a ``finally`` (also on failure).
    """

    if source.is_file():
        cap = cv2.VideoCapture(str(source))
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open reference video: {source}")
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield f"{source}:{index}", frame
                index += 1
        finally:
            cap.release()  # also covers the never-opened case
    else:
        for path in frame_paths(source):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Undecodable frame: {path}")
            yield path.name, frame


def frame_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    """PSNR (dB, 8-bit scale; ``null`` when bit-identical) and MAE."""

    if actual.shape != reference.shape:
        raise ValueError(
            f"Frame geometry mismatch: {actual.shape} vs {reference.shape}"
        )
    if actual.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError("Frames must be uint8 BGR")
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    mae = float(np.abs(diff).mean())
    mse = float(np.mean(diff * diff))
    psnr = None if mse == 0 else 10.0 * math.log10(255.0 * 255.0 / mse)
    return {"psnr_db": psnr, "mae": mae}


def _round(value):
    return None if value is None else round(value, 4)


def compare(actual_dir: Path, reference: Path, limit: int | None) -> dict:
    """Stream both sides in lockstep and build the full comparison record."""

    actual_stream = stream_frames(actual_dir)
    reference_stream = stream_frames(reference)
    prefix = limit is not None
    wanted = limit if prefix else None
    samples = []
    perfect = 0
    actual_extra = reference_extra = 0
    try:
        while True:
            if wanted is not None and len(samples) >= wanted:
                # A prefix check stops after N pairs; the remaining frames
                # are intentionally not read, so exhaustiveness beyond N is
                # unknown — but both sides must have been able to provide N.
                break
            actual_item = next(actual_stream, None)
            reference_item = next(reference_stream, None)
            if actual_item is not None and reference_item is not None:
                label, actual_frame = actual_item
                _, reference_frame = reference_item
                metrics = frame_metrics(actual_frame, reference_frame)
                if metrics["psnr_db"] is None:
                    perfect += 1
                metrics["frame"] = len(samples)
                metrics["source"] = label
                samples.append(metrics)
                continue
            # At least one side ended; count what the other still holds.
            if actual_item is not None:
                actual_extra += 1
                for _ in actual_stream:
                    actual_extra += 1
            if reference_item is not None:
                reference_extra += 1
                for _ in reference_stream:
                    reference_extra += 1
            break
    finally:
        # Deterministic release on every exit (short limit, consumer error,
        # normal end): closing a generator runs its VideoCapture finally.
        actual_stream.close()
        reference_stream.close()

    if not samples:
        raise ValueError("Both sides need at least one frame")
    actual_count = len(samples) + actual_extra
    reference_count = len(samples) + reference_extra
    counts_equal = actual_extra == 0 and reference_extra == 0
    short_for_limit = (
        prefix and (actual_count < limit or reference_count < limit)
    )
    if not prefix and not counts_equal:
        print(
            f"FAIL: frame counts differ (actual {actual_count} vs reference "
            f"{reference_count}); the default full comparison requires equal "
            "counts — use --limit N for a labeled prefix check",
            file=sys.stderr,
        )
    if short_for_limit:
        print(
            f"FAIL: --limit {limit} needs at least {limit} frames on both "
            f"sides (actual {actual_count}, reference {reference_count}); "
            "it verifies the advertised prefix, never a shorter one",
            file=sys.stderr,
        )
    # Aggregates come from the raw metrics; rounding happens only on the
    # values written out below.
    real_psnr = [item["psnr_db"] for item in samples if item["psnr_db"] is not None]
    all_perfect = perfect == len(samples)
    maes = [item["mae"] for item in samples]
    record = {
        "actual": str(actual_dir),
        "reference": str(reference),
        "frames_compared": len(samples),
        "actual_frame_count": actual_count,
        "reference_frame_count": reference_count,
        "counts_equal": counts_equal,
        "prefix_check": prefix,
        "limit": limit,
        "short_for_limit": short_for_limit,
        "perfect_frames": perfect,
        "mean_psnr_db": None
        if all_perfect
        else round(float(np.mean(real_psnr)), 4),
        "mean_mae": round(float(np.mean(maes)), 4),
        "min_psnr_db": None if all_perfect else round(float(np.min(real_psnr)), 4),
        "max_mae": round(float(np.max(maes)), 4),
        "mean_psnr_note": "null means every compared frame is bit-identical "
        "(no finite PSNR); otherwise the mean covers the frames with a "
        "real MSE",
        "frames": [
            {
                "frame": item["frame"],
                "source": item["source"],
                "psnr_db": _round(item["psnr_db"]),
                "mae": round(item["mae"], 4),
            }
            for item in samples
        ],
    }
    record["count_mismatch"] = not counts_equal
    return record


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True, help="rendered frame directory")
    parser.add_argument(
        "--reference",
        required=True,
        help="reference frame directory or video file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="labeled prefix check: compare only the first N frames "
        "(N > 0); the default full comparison fails on unequal counts",
    )
    parser.add_argument(
        "--gate", type=float, help="minimum required mean PSNR in dB (finite)"
    )
    parser.add_argument("--output", type=Path, help="write the JSON record here")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error(f"--limit must be a positive frame count, got {args.limit}")
    if args.gate is not None and not math.isfinite(args.gate):
        parser.error(f"--gate must be a finite dB value, got {args.gate}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        record = compare(Path(args.actual), Path(args.reference), args.limit)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    mean = record["mean_psnr_db"]
    print(
        f"frames={record['frames_compared']} "
        f"perfect={record['perfect_frames']} mean_psnr_db={mean} "
        f"mean_mae={record['mean_mae']} min_psnr_db={record['min_psnr_db']} "
        f"counts_equal={record['counts_equal']}"
        + (" prefix_check" if record["prefix_check"] else "")
    )
    if not record["prefix_check"] and record["count_mismatch"]:
        return 1
    if record.get("short_for_limit"):
        return 1
    if args.gate is not None:
        if mean is None or mean + 1e-9 >= args.gate:
            # A fully perfect comparison passes any finite gate.
            print(f"PASS: mean PSNR {'(perfect)' if mean is None else mean} "
                  f">= gate {args.gate} dB")
        else:
            print(
                f"FAIL: mean PSNR {mean} dB < gate {args.gate} dB",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
