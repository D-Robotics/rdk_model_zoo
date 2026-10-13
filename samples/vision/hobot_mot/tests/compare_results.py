#!/usr/bin/env python3
"""Compare result JSONs from the HobotMot sample runtimes and the reference.

The files come from runtime/python main.py --output, runtime/cpp
hobot_mot_replay --output and tests/cpp parity_reference --output, all run on
the same board with the same input JSONL and config. Identities, boxes and
states must match exactly (same upstream instances, same input order); only
the score may differ by float formatting (jsoncpp vs Python json), bounded
here by 1e-6.

Usage: python3 compare_results.py FILE FILE [FILE ...]
Exit status 0 when every file pair matches, 1 otherwise.
"""
from __future__ import annotations

import json
import sys

SCORE_TOLERANCE = 1e-6


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON array of frames")
    return data


def frame_signature(frame: dict) -> tuple:
    """Identity of a frame ignoring nothing: index, ts, rows, removed ids.

    Track rows keep input order (one row per input detection); removed ids
    are compared order-insensitively because their cross-class order is an
    iteration artifact, not tracking semantics.
    """
    rows = tuple(
        (row["class_id"], row["track_id"], tuple(row["bbox"]), row["state"])
        for row in frame["tracks"])
    gone = tuple(sorted(
        (row["class_id"], row["track_id"])
        for row in frame["disappeared"]))
    return frame["frame_index"], frame["timestamp_ms"], rows, gone


def score_vector(frames: list[dict]) -> list[float]:
    return [row["score"] for frame in frames for row in frame["tracks"]]


def describe_first_difference(base: list[dict], other: list[dict],
                              base_name: str, other_name: str) -> str | None:
    if len(base) != len(other):
        return (f"frame count {len(base)} vs {len(other)}")
    for index, (base_frame, other_frame) in enumerate(zip(base, other)):
        base_sig = frame_signature(base_frame)
        other_sig = frame_signature(other_frame)
        if base_sig == other_sig:
            continue
        if base_sig[0] != other_sig[0] or base_sig[1] != other_sig[1]:
            return (f"frame {index}: header {base_sig[:2]} vs {other_sig[:2]}")
        for row_index, (base_row, other_row) in enumerate(
                zip(base_sig[2], other_sig[2])):
            if base_row != other_row:
                return (f"frame {index} track {row_index}: "
                        f"{base_row} vs {other_row}")
        return (f"frame {index}: disappeared {base_sig[3]} vs {other_sig[3]}")
    return None


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    paths = argv[1:]
    frames = {path: load(path) for path in paths}
    base = paths[0]
    failures = 0
    for path in paths[1:]:
        difference = describe_first_difference(frames[base], frames[path],
                                               base, path)
        if difference:
            print(f"MISMATCH {base} vs {path}: {difference}")
            failures += 1
            continue
        base_scores = score_vector(frames[base])
        other_scores = score_vector(frames[path])
        worst = max((abs(a - b) for a, b in zip(base_scores, other_scores)),
                    default=0.0)
        if worst > SCORE_TOLERANCE:
            print(f"MISMATCH {base} vs {path}: score difference {worst} "
                  f"exceeds {SCORE_TOLERANCE}")
            failures += 1
            continue
        print(f"MATCH {base} vs {path}: {len(frames[path])} frames, "
              f"{len(other_scores)} rows, max score delta {worst:g}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
