# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Depth CLI: arguments, image IO, scheduling, visualization and provenance."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.vision.depth_anything_v2.runtime.python.model_binding import (
    SAMPLE_DIR,
    resolve_selection,
    list_available_assets,
)
from samples.vision.depth_anything_v2.runtime.python.model_runner import (
    RuntimeModelRunner,
)


def build_parser():
    p = argparse.ArgumentParser(
        description="Depth Anything V2 relative depth; published S100 artifact only"
    )
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument(
        "--test-img", type=Path, default=SAMPLE_DIR / "test_data/furseal.jpg"
    )
    p.add_argument("--output", type=Path, default=Path("outputs/depth_anything_v2"))
    p.add_argument(
        "--img-save-path",
        type=Path,
        help="Optional additional source-compatible color image path; must not exist",
    )
    p.add_argument("--resize-type", type=int, choices=(0, 1), default=0)
    p.add_argument("--priority", type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
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
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        if not 0 <= args.priority <= 255 or any(core < 0 for core in args.bpu_cores):
            raise ValueError("Priority must be 0..255 and BPU cores nonnegative")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "model_path": str(selection.model_path),
                        "resize_type": args.resize_type,
                        "sdk_loaded": False,
                        "downloaded": False,
                    },
                    indent=2,
                )
            )
            return 0
        output = args.output.expanduser()
        extra = args.img_save_path.expanduser() if args.img_save_path else None
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        reserved = {
            output / name
            for name in (
                "raw_depth.npy",
                "depth_native.npy",
                "depth_gray.png",
                "depth_color.png",
                "report.json",
            )
        }
        if extra is not None and extra.resolve() in {p.resolve() for p in reserved}:
            raise ValueError(
                "Additional color image must not replace a canonical output"
            )
        if extra is not None and extra.exists():
            raise FileExistsError(f"Use a new image path: {extra}")
        import cv2
        import numpy as np
        from samples._shared.assets import sha256_file
        from samples._shared.runtime_meta import metadata_evidence
        from samples.vision.depth_anything_v2.runtime.python.depth_anything_v2 import (
            DepthAnythingV2Task,
        )
        from samples.vision.depth_anything_v2.runtime.python.visualization import (
            normalize_depth,
            colorize_depth,
        )

        image_path = args.test_img.expanduser()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode input image: {image_path}")
        runner = RuntimeModelRunner(selection)
        binding = runner.load()
        runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
        task = DepthAnythingV2Task(runner, binding, resize_type=args.resize_type)
        prepared = task.pre_process(image)
        raw = task.forward(prepared.tensors)
        result = task.post_process(raw, prepared.context)
        color = colorize_depth(result.depth_native)
        report = {
            "schema_version": "1.0",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "model_path": str(selection.model_path),
            "model_sha256": sha256_file(selection.model_path),
            "publisher_sha256": selection.asset.sha256,
            "input": str(image_path),
            "input_sha256": sha256_file(image_path),
            "runtime_version": str(getattr(runner.runtime, "version", "unknown")),
            "runtime_metadata": metadata_evidence(binding.metadata),
            "resize_type": args.resize_type,
            "input_normalization": "pixelwise RGB z-score, epsilon=1e-5",
            "output_units": "relative, not meters",
            "depth_native_shape": list(result.depth_native.shape),
            "priority": args.priority,
            "bpu_cores": args.bpu_cores,
            "restoration": "OpenCV INTER_LINEAR; optional letterbox crop before original-size resize",
            "constant_visualization": "all zero grayscale",
            "latency": "not measured",
            "additional_color_image": str(extra) if extra else None,
        }
        output.mkdir(parents=True, exist_ok=False)
        np.save(output / "raw_depth.npy", raw)
        np.save(output / "depth_native.npy", result.depth_native)
        if not cv2.imwrite(
            str(output / "depth_gray.png"), normalize_depth(result.depth_native)
        ) or not cv2.imwrite(str(output / "depth_color.png"), color):
            raise OSError("Failed to write depth visualization")
        if extra is not None:
            extra.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(extra), color):
                raise OSError(f"Failed to write {extra}")
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved relative depth and provenance to {output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
