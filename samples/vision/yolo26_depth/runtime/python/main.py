# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI: preparation, measured forward call, rendering and evidence IO."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.yolo26_depth.runtime.python.model_binding import (
    SAMPLE_DIR,
    TARGETS,
    VARIANTS,
    resolve_selection,
    list_available_assets,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="YOLO26 relative depth on X5/S100/S100P/S600"
    )
    p.add_argument("--target", choices=("auto",) + TARGETS, default="auto")
    p.add_argument(
        "--variant",
        choices=VARIANTS,
        help="Default n; exact asset-id may infer another variant",
    )
    p.add_argument("--asset-id")
    p.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=Path,
        help="External published asset path; requires exact asset-id",
    )
    p.add_argument(
        "--converted-model",
        action="store_true",
        help="Explicit custom artifact using the asset-id tensor contract; no publisher hash claim",
    )
    p.add_argument(
        "--test-img",
        "--input",
        dest="test_img",
        type=Path,
        default=SAMPLE_DIR / "test_data/bus.jpg",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/yolo26_depth"),
        help="New directory; existing paths are refused",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--priority",
        type=int,
        default=None,
        help="S default 0; X5 leaves SDK default unless set",
    )
    p.add_argument(
        "--bpu-cores",
        type=int,
        nargs="+",
        default=None,
        help="S default [0]; X5 leaves SDK default unless set",
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--list-models", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {"asset_id": a.reference, "url": a.url, "sha256": a.sha256}
                        for a in list_available_assets(args.target)
                    ],
                    indent=2,
                )
            )
            return 0
        selection = resolve_selection(
            args.target,
            variant=args.variant,
            asset_id=args.asset_id,
            model_path=args.model_path,
            converted_model=args.converted_model,
        )
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        if args.priority is not None and not 0 <= args.priority <= 255:
            raise ValueError("priority must be 0..255")
        if args.bpu_cores is not None and any(i < 0 for i in args.bpu_cores):
            raise ValueError("bpu-cores must be nonnegative")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "variant": selection.variant,
                        "profile": selection.profile,
                        "asset_id": (
                            None
                            if selection.converted_model
                            else selection.asset.reference
                        ),
                        "contract_reference": selection.asset.reference,
                        "artifact_origin": (
                            "user-converted"
                            if selection.converted_model
                            else "published-manifest"
                        ),
                        "model_path": str(selection.model_path),
                        "output_semantics": (
                            "raw_logit"
                            if selection.profile == "lite"
                            else "calibrated_log_depth"
                        ),
                        "sdk_loaded": False,
                        "downloaded": False,
                    },
                    indent=2,
                )
            )
            return 0
        output = args.output.expanduser()
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        import cv2
        import numpy as np
        from samples._shared.assets import sha256_file
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.yolo26_depth.runtime.python.model_runner import (
            RuntimeModelRunner,
        )
        from samples.vision.yolo26_depth.runtime.python.yolo26_depth import (
            Yolo26DepthTask,
        )
        from samples.vision.yolo26_depth.runtime.python.visualization import (
            colorize_depth,
        )

        image_path = args.test_img.expanduser()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode input image: {image_path}")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        priority = (
            args.priority
            if args.priority is not None
            else (None if selection.target == "x5" else 0)
        )
        cores = (
            args.bpu_cores
            if args.bpu_cores is not None
            else (None if selection.target == "x5" else [0])
        )
        runner.set_scheduling_params(priority=priority, bpu_cores=cores)
        task = Yolo26DepthTask(runner, binding)
        prepared = task.pre_process(image)
        for _ in range(args.warmup):
            task.forward(prepared.tensors)
        started = time.perf_counter()
        raw = task.forward(prepared.tensors)
        elapsed = (time.perf_counter() - started) * 1000
        result = task.post_process(raw, prepared.context)
        color = colorize_depth(result.depth_native)
        overlay = cv2.addWeighted(image, 0.45, color, 0.55, 0.0)
        report = {
            "schema_version": "1.0",
            "target": selection.target,
            "variant": selection.variant,
            "profile": selection.profile,
            "asset_id": (
                None if selection.converted_model else selection.asset.reference
            ),
            "contract_reference": selection.asset.reference,
            "artifact_origin": (
                "user-converted" if selection.converted_model else "published-manifest"
            ),
            "model_path": str(selection.model_path),
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": (
                None if selection.converted_model else selection.asset.sha256
            ),
            "input": str(image_path),
            "input_sha256": sha256_file(image_path),
            "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
            "metadata": metadata_evidence(binding.metadata),
            "log_depth_shape": list(result.log_depth.shape),
            "depth_native_shape": list(result.depth_native.shape),
            "warmup": args.warmup,
            "latency_ms": elapsed,
            "latency_scope": "one forward including transport validation and owned output copy; not BPU-only",
            "priority": priority,
            "bpu_cores": cores,
            "geometry": vars(result.context),
            "depth_units": "relative; not calibrated metres",
        }
        if result.raw_logit is not None:
            from samples.vision.yolo26_depth.runtime.python.model_binding import (
                LITE_CALIBRATION,
            )

            a, b = LITE_CALIBRATION[selection.variant]
            report["calibration"] = {"cal_a": a, "cal_b": b, "clip": [-4, 5]}
            report["raw_logit_shape"] = list(result.raw_logit.shape)
        output.mkdir(parents=True, exist_ok=False)
        np.save(output / "log_depth.npy", result.log_depth, allow_pickle=False)
        np.save(output / "depth_native.npy", result.depth_native, allow_pickle=False)
        if result.raw_logit is not None:
            np.save(output / "raw_logit.npy", result.raw_logit, allow_pickle=False)
        for name, value in (("depth.png", color), ("overlay.png", overlay)):
            if not cv2.imwrite(str(output / name), value):
                raise OSError(f"Could not write {output/name}")
        text = json.dumps(report, indent=2)
        (output / "report.json").write_text(text + "\n")
        print(text)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
