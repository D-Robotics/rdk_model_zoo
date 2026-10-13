"""Visible entry point for the HobotMot Python replay.

The entry does four things in the open: parse arguments, construct the named
tracker, call ``predict`` per frame and deliver the results. Loading,
rendering and saving helpers live in ``cli.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Runnable three ways: as a script (run.sh puts this directory on sys.path
# itself), by file path (the sample-contract checker imports main.py
# directly), and from a standalone copy of the sample directory on a board.
# The bootstrap makes the sibling imports resolve in the last two cases.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cli import (  # noqa: E402
    build_parser, load_frame_records, render_frames, save_results)
from track import HobotMotTracker  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_frame_records(args.input)
        tracker = HobotMotTracker(config_path=args.config,
                                  frame_width=args.frame_width,
                                  frame_height=args.frame_height,
                                  native_lib=args.native_lib)
        with tracker:
            results = [
                tracker.predict(record.detections, record.timestamp_ms,
                                record.image_width or None,
                                record.image_height or None)
                for record in records
            ]
    # MotConfigError and MotInputError are ValueError subclasses; the
    # native binding raises RuntimeError. All become a readable error and
    # exit code 2 at this CLI boundary.
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    render_frames(results, records)
    if args.output:
        save_results(args.output, records, results)
        print(f"results written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
