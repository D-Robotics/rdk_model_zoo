# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""ByteTrack CLI options, detector-asset selection, listing, and dry-run.

``main.py`` uses these helpers to parse arguments and preview a selection.
The tracking flow lives in ``tracking.py``; track overlay rendering lives
in ``cli.py``. ByteTrack owns its published detector identities;
the tensor math is shared with YOLOv5.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from samples.vision.yolov5.runtime.python import cli as detector
from utils.py_utils.platforms import resolve_target

SAMPLE_DIR = Path(__file__).resolve().parents[2]


def resolve_selection(target='auto', **kwargs):
    """Resolve ByteTrack's exact detector assets for x5 and s100/s100p/s600.

    X5 reuses the published YOLOv5 asset identities (``x5:yolov5:*`` .bin):
    those rows are served through the shared catalog's yolov5 consumer (the
    selection keeps that publisher identity), and a catalog-default local
    copy is rebased under this sample's ``model/`` directory; an explicit
    ``model_path`` argument passes through unchanged. S targets resolve
    through ByteTrack's own manifest rows directly.

    Args:
        target: ``auto`` resolves the executing board; x5 and S targets publish.
        **kwargs: Optional variant/asset-id/model-path filters forwarded to
            the shared YOLOv5 resolver.

    Returns:
        ModelSelection: Concrete target, manifest asset, and local path
        (always under this sample's ``model/`` directory unless model-path
        overrides it).

    Raises:
        ValueError: The target or asset combination is invalid.
    """
    resolved = resolve_target(target, soc_name=kwargs.get('soc_name'),
                              board_type=kwargs.get('board_type'))
    consumer = 'yolov5' if resolved == 'x5' else 'bytetrack'
    selection = detector.resolve_selection(target, consumer=consumer, **kwargs)
    if consumer == 'yolov5' and kwargs.get('model_path') is None:
        # Keep the yolov5 consumer identity (the asset's publisher — the
        # detector's binding re-validates the reference against that
        # consumer's rows) while managing the catalog-default local copy
        # under this sample. An explicit model_path already returned by the
        # shared resolver is the caller's choice and stays untouched.
        selection = replace(selection,
                            model_path=SAMPLE_DIR / 'model' / selection.asset.filename)
    return selection


def list_available_assets(target=None):
    """List supported detector assets, without hardware or network access.

    Args:
        target: Concrete target filter; ``auto``/None lists all publications.

    Returns:
        The exact asset rows this sample consumes. X5 rows reuse the YOLOv5
        publication identity; S rows are ByteTrack's own.
    """
    x5_rows = detector.list_available_assets('x5', consumer='yolov5') \
        if target in (None, 'auto', 'x5') else ()
    s_rows = detector.list_available_assets(target, consumer='bytetrack') \
        if target in (None, 'auto', 's100', 's100p', 's600') else ()
    return tuple(x5_rows) + tuple(s_rows)


def build_parser():
    """Build the prepared-video ByteTrack parser with source defaults.

    Returns:
        argparse.ArgumentParser: Parser for selection, video IO, tracking,
        scheduling, and model-free listing/dry-run options.
    """
    p = argparse.ArgumentParser(
        description='Explicit prepared-video ByteTrack CLI; no automatic model/video installation.')
    p.add_argument('--target', default='auto', choices=('auto', 'x5', 's100', 's100p', 's600'))
    p.add_argument('--variant',
                   help='X5: n/s/m/l/x-v7.0 or s/m/l/x-v2.0; S: x-672. Omitted uses n-v7.0 / x-672.')
    p.add_argument('--asset-id')
    p.add_argument('--model-path')
    p.add_argument('--input', default=str(SAMPLE_DIR / 'test_data/track_test.mp4'),
                   help='User-prepared video, not bundled or downloaded by this entry.')
    p.add_argument('--output', default=str(SAMPLE_DIR / 'test_data/result_unified.mp4'))
    p.add_argument('--records', default=None,
                   help='Optional JSONL with every frame and its owned track records.')
    p.add_argument('--score-thres', type=float, default=.25)
    p.add_argument('--nms-thres', type=float, default=.45)
    p.add_argument('--track-thresh', type=float, default=.3)
    p.add_argument('--track-buffer', type=int, default=60)
    p.add_argument('--match-thresh', type=float, default=.8)
    p.add_argument('--frame-rate', type=int, default=30)
    p.add_argument('--mot20', action='store_true')
    p.add_argument('--priority', type=int, default=0)
    p.add_argument('--bpu-cores', type=int, nargs='+', default=[0])
    p.add_argument('--max-frames', type=int, default=0, help='0 processes all video frames.')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--list-models', action='store_true')
    modes.add_argument('--dry-run', action='store_true')
    return p


def run_list_models(target) -> int:
    """Print ByteTrack's exact detector asset references (model-free).

    Args:
        target: Concrete target or ``auto``.

    Returns:
        int: 0 after printing the list.
    """
    for asset in list_available_assets(target):
        print(asset.reference)
    return 0


def run_dry_run(selected, cfg, input_path: str) -> int:
    """Print the resolved selection and tracking defaults without loading.

    Args:
        selected: Resolved detector selection to preview.
        cfg: TrackingConfig whose source defaults are recorded.
        input_path: Prepared video path recorded in the preview.

    Returns:
        int: 0 after printing the preview.
    """
    print(json.dumps(dict(target=selected.target, variant=selected.variant,
                          asset_id=selected.asset.reference,
                          model_path=str(selected.model_path), input=input_path,
                          video_status='user preparation required', board_status='not-run',
                          tracking=asdict(cfg)), indent=2))
    return 0


def draw_tracks(image,tracks):
    """Draw track IDs and boxes on an owned copy of the BGR image.

    Args:
        image: Original uint8 BGR image.
        tracks: Track objects with original-pixel tlbr boxes and integer IDs.

    Returns:
        BGR image with the same shape, with source ID colors and labels.
    """
    import cv2

    result=image.copy()
    for t in tracks:
        x1,y1,x2,y2=(int(x) for x in t.tlbr);tid=t.track_id
        color=((37*tid)%255,(17*tid)%255,(29*tid)%255)
        cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
        cv2.putText(result,f'ID:{tid}',(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    return result
