# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free DINOv2 CLI and board execution entrypoint.

Option declarations, the model-free listing/dry-run modes and the feature
summary/export helpers live in ``cli.py``.  This entry stays focused on the
execution path: resolve the selection, load the runner, construct
``DINOv2Task`` and call ``predict`` — once for the primary image and once more
for the optional similarity image.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.dinov2.runtime.python.cli import (  # noqa: E402
    build_parser,  # re-exported here: the contract checker imports it from main
    cosine,
    read_bgr_image,
    run_dry_run,
    run_list_models,
    save_feature,
    summary,
)
from samples.vision.dinov2.runtime.python.cli import resolve_selection  # noqa: E402


def main(argv=None) -> int:
    """Resolve contracts or run one image through the real task pipeline."""

    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            return run_list_models(args.target)
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires --target s100, s100p, or s600; no board detection performed.")
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if args.dry_run:
            return run_dry_run(selection)
        from utils.py_utils.platforms import require_execution_target

        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {selection.model_path}; prepare it explicitly with model/download.sh.")
        from samples.vision.dinov2.runtime.python.embedding import DINOv2Embedder

        image = read_bgr_image(args.test_img)
        task = DINOv2Embedder(selection, output=args.output)
        task.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        feature_a = task.predict(image)
        report = summary(feature_a, args.output)

        second_path = Path(args.second_img).expanduser()
        if second_path.is_file():
            feature_b = task.predict(read_bgr_image(second_path))
            similarity = cosine(feature_a, feature_b)
            report["second_image"] = {"path": str(second_path), "status": "used"}
            report["cosine_similarity"] = similarity
            if similarity is None:
                report["cosine_status"] = "skipped_zero_norm"
        else:
            report["second_image"] = {"path": str(second_path), "status": "skipped_missing"}
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        if args.output_file is not None:
            save_feature(Path(args.output_file).expanduser(), feature_a)
        return 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
