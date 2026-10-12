# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""LiteAvatar entry: parse, select, construct, predict, deliver the video.

This file stays deliberately small and keeps the flow visible: parse the
arguments, validate them, handle the model-free listing/dry-run modes, gate
the board, construct the four-model :class:`LiteAvatarPipeline`, call
``pipeline.predict`` once for the whole clip, then hand the result to
:func:`deliver_frames` for video delivery.  Option declarations, published
selection, the writers and the delivery loop live in ``cli.py``; the stage
composition lives in ``avatar.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.lite_avatar.runtime.python.cli import (  # noqa: E402
    build_parser,
    deliver_frames,
    make_writer,
    missing_models,
    normalize_args,
    print_model_list,
    print_resolution,
    print_result,
    read_audio,
    resolve_encoder,
    resolve_selections_for,
)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        normalize_args(args)
        if args.list_models:
            # Listing is host-independent: an auto target lists the only
            # published set instead of probing for a board.
            return print_model_list("s100" if args.target == "auto" else args.target)
        selections = resolve_selections_for(args)
        if args.dry_run:
            return print_resolution(args, selections)
        missing = missing_models(selections)
        if missing:
            print("Missing model artifacts (prepare per model/README.md):", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
            return 2

        # A real execution proves the exact board before the SDK is imported;
        # the stage runners repeat this check immediately before loading.
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selections[0].target)

        # Thread policy of the measured board configuration (the source
        # deployment's values): the BLAS/OpenMP caps must land before the
        # first NumPy/Torch import below, and Torch/OpenCV thread pools are
        # bounded to 2 so the Python renderer, the BPU runtime and the
        # encoder worker stay off each other's cores.  ``setdefault`` keeps
        # an explicitly exported environment in control, and none of this
        # runs on import of the pipeline API — only this entry sets it.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        import cv2
        import torch

        torch.set_num_threads(2)
        cv2.setNumThreads(2)

        # Imported inside real execution: Torch/torchaudio (frontend and
        # interpolation), scipy (smoothing) and hbm_runtime (models) load
        # only after the selection and board checks passed.
        from samples.vision.lite_avatar.runtime.python.avatar import (
            AvatarInput,
            LiteAvatarPipeline,
        )

        pipeline = LiteAvatarPipeline.from_models(
            selections, args.cmvn_path, args.avatar_data
        )
        if args.priority is not None or args.bpu_cores is not None:
            pipeline.set_scheduling_params(
                priority=args.priority, bpu_cores=args.bpu_cores
            )
        waveform, sample_rate = read_audio(args.audio)
        result = pipeline.predict(AvatarInput(waveform, sample_rate))
        encoder_kind = resolve_encoder(args.encoder, selections[0].target)
        writer = None
        if not args.no_encode:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = make_writer(
                encoder_kind, args.output, args.audio, result.width, result.height
            )
        seconds = deliver_frames(result, writer)
        print_result(
            result, args.output, encoder_kind, seconds, writer is not None,
            pipeline.last_timings,
        )
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
