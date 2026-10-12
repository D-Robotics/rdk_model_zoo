# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run the Faster R-CNN command-line sample.

Parse options, construct the paired runtime and the detection task (or the
person tracker for video), call ``predict``, and report the result as JSON.
Selection, listing, dry-run, labels, drawing and the image/video IO loops
live in ``cli.py``; the pipeline stages live in ``detect.py`` and
``tracking.py``.  Help, listing and dry-run stay model-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.fasterrcnn.runtime.python.cli import (  # noqa: F401 - build_parser re-exported for the contract checker
    build_parser,
)
from samples.vision.fasterrcnn.runtime.python.cli import (
    BindingError,
    dry_run,
    list_models,
    require_pair_files,
    resolve_selection,
    save_detection_image,
    track_video,
    verify_pair_hashes,
)


def main(argv=None) -> int:
    """Run list, dry-run, image inference or person video tracking.

    Returns:
        int: 0 on success; 2 for a reported selection, option, IO or
        runtime error.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return list_models(args.target)
        selection = resolve_selection(args.target, parta_path=args.parta_path,
                                      partb_path=args.partb_path)
        if args.dry_run:
            return dry_run(selection, args)
        require_pair_files(selection)
        hashes = verify_pair_hashes(selection)

        import cv2

        if args.threads is not None:
            cv2.setNumThreads(args.threads)

        from samples.vision.fasterrcnn.runtime.python.detect import FasterRCNNTask

        task = FasterRCNNTask(selection)
        if args.track or args.video is not None:
            from samples.vision.fasterrcnn.runtime.python.tracking import PersonTrackerTask

            video_path = Path(args.video) if args.video else Path(args.input)
            out_path = Path(args.save_path).with_suffix(".mp4")
            summary = track_video(PersonTrackerTask(task), video_path, out_path,
                                  reset_every=args.reset_every,
                                  records_path=args.records,
                                  encode=not args.no_encode)
            print(json.dumps({"asset_id": selection.asset_id, **hashes,
                              **summary}))
            return 0

        image = cv2.imread(str(Path(args.input).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"input image not found or unreadable: {args.input}")
        import time

        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        result = task.predict(image)
        timing = {
            "predict_ms": (time.perf_counter() - wall_started) * 1e3,
            "cpu_ms": (time.process_time() - cpu_started) * 1e3,
            "proposals": task.last_proposals,
        }
        report = save_detection_image(image, result, args.label_file, args.save_path)
        print(json.dumps({"asset_id": selection.asset_id, **hashes,
                          **timing, **report}))
        return 0
    except (BindingError, FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
