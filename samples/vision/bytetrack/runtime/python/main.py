# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""People and Agent entrypoint for the ByteTrack tracking sample.

This file stays deliberately small: parse the arguments, resolve the
detector, construct the tracker through ``ByteTrackTask.from_model``, call
``predict`` per frame, write the annotated video. Option declarations,
model-free listing/dry-run live in ``cli.py``; the tracking flow lives in
``tracking.py``; overlay rendering lives in ``cli.py``.
"""
import json, math, sys
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.bytetrack.runtime.python.cli import (  # noqa: E402
    build_parser, resolve_selection, run_dry_run, run_list_models,
)


def main(argv=None):
    a = build_parser().parse_args(argv)
    capture = writer = record_file = None
    try:
        if a.list_models:
            return run_list_models(a.target)
        if a.dry_run and a.target == 'auto':
            raise ValueError('--dry-run requires explicit --target.')
        from samples.vision.bytetrack.runtime.python.tracking import TrackingConfig
        cfg = TrackingConfig(a.track_thresh, a.track_buffer, a.match_thresh, a.frame_rate, a.mot20)
        if a.max_frames < 0 or not 0 <= a.priority <= 255 or any(x < 0 for x in a.bpu_cores):
            raise ValueError('Invalid max-frames or scheduling parameters.')
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in [a.score_thres, a.nms_thres]):
            raise ValueError('Detection thresholds must be finite [0,1].')
        selected = resolve_selection(a.target, model_path=a.model_path, asset_id=a.asset_id)
        if a.dry_run:
            return run_dry_run(selected, cfg, a.input)
        import cv2
        from samples.vision.bytetrack.runtime.python.tracking import ByteTrackTask
        from samples.vision.bytetrack.runtime.python.cli import draw_tracks

        input_path = Path(a.input).expanduser()
        output_path = Path(a.output).expanduser()
        if not input_path.is_file():
            raise ValueError(f'Missing video: {input_path}; prepare it explicitly first.')
        if input_path.resolve() == output_path.resolve():
            raise ValueError('Output video must differ from input video.')
        if a.records and Path(a.records).expanduser().resolve() in (input_path.resolve(), output_path.resolve()):
            raise ValueError('records must differ from video paths.')

        # Real execution starts here: from_model constructs and loads the
        # detector; the entry never creates runners or bindings itself.
        task = ByteTrackTask.from_model(selected, config=cfg,
                                        score_thres=a.score_thres, nms_thres=a.nms_thres)
        task.detector.set_scheduling_params(priority=a.priority, bpu_cores=a.bpu_cores)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {input_path}')
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise OSError(f'Cannot open video writer: {output_path}')
        if a.records:
            record_path = Path(a.records).expanduser()
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_file = record_path.open('w')
        count = 0
        while not a.max_frames or count < a.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            tracks = task.predict(frame)
            writer.write(draw_tracks(frame, tracks))
            count += 1
            if record_file:
                record_file.write(json.dumps(dict(frame=count, tracks=[asdict(t) for t in tracks])) + '\n')
        if not count:
            raise ValueError('Video contained no decodable frames.')
        print(f'Saved {count} tracked frames to {output_path}')
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if record_file is not None:
            record_file.close()


if __name__ == '__main__':
    raise SystemExit(main())
