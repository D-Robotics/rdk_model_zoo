# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free DINOv2 CLI and board execution entrypoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples.vision.dinov2.runtime.python.model_binding import (  # noqa: E402
    OUTPUT_SHAPES,
    SAMPLE_DIR,
    SUPPORTED_TARGETS,
    list_available_assets,
    resolve_selection,
)

DEFAULT_TEST_IMAGE = SAMPLE_DIR / "test_data/dog.jpg"
DEFAULT_SECOND_IMAGE = SAMPLE_DIR / "test_data/bus.jpg"


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without reading board identity or importing SDKs."""

    parser = argparse.ArgumentParser(description="DINOv2 ViT-S/14 image embedding")
    parser.add_argument("--target", choices=("auto", *SUPPORTED_TARGETS), default="auto", help="Concrete execution target.")
    parser.add_argument("--asset-id", default=None, help="Exact qualified manifest asset reference.")
    parser.add_argument("--model-path", default=None, help="Explicit local HBM path; requires --asset-id.")
    parser.add_argument("--test-img", default=str(DEFAULT_TEST_IMAGE), help="First BGR image path.")
    parser.add_argument("--second-img", default=str(DEFAULT_SECOND_IMAGE), help="Optional second image for cosine similarity.")
    parser.add_argument("--output", choices=("cls_feat", "patch_feat"), default="cls_feat", help="Feature output to return.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority 0..255.")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0], help="Nonnegative BPU core indexes.")
    parser.add_argument("--output-file", default=None, help="Optional exact path for the returned NumPy tensor.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List exact manifest assets without loading a model.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve one target and print its source contract without SDK loading.")
    return parser


def _summary(tensor, output: str) -> dict:
    flat = tensor.reshape(-1).astype("float32")
    return {
        "output": output,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float((flat ** 2).sum() ** 0.5),
    }


def _cosine(first, second):
    import numpy as np

    a = first.reshape(-1).astype(np.float64)
    b = second.reshape(-1).astype(np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(a @ b / (norm_a * norm_b))


def _run(selection, args) -> int:
    import cv2
    import numpy as np

    from samples.vision.dinov2.runtime.python.embedding import DINOv2Task
    from samples.vision.dinov2.runtime.python.model_runner import RuntimeModelRunner

    image = cv2.imread(str(Path(args.test_img).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {args.test_img}")
    runner = RuntimeModelRunner(selection)
    binding = runner.load()
    runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    task = DINOv2Task(runner, binding, args.output)
    feature_a = task.predict(image)
    report = _summary(feature_a, args.output)

    second_path = Path(args.second_img).expanduser()
    if second_path.is_file():
        image_b = cv2.imread(str(second_path), cv2.IMREAD_COLOR)
        if image_b is None:
            raise ValueError(f"Cannot read image: {args.second_img}")
        feature_b = task.predict(image_b)
        cosine = _cosine(feature_a, feature_b)
        report["second_image"] = {"path": str(second_path), "status": "used"}
        report["cosine_similarity"] = cosine
        if cosine is None:
            report["cosine_status"] = "skipped_zero_norm"
    else:
        report["second_image"] = {"path": str(second_path), "status": "skipped_missing"}
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if args.output_file is not None:
        output_path = Path(args.output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            np.save(handle, feature_a, allow_pickle=False)
        print(f"Feature tensor saved: {output_path}")
    return 0


def main(argv=None) -> int:
    """Resolve contracts or run one image through the real task pipeline."""

    args = build_parser().parse_args(argv)
    try:
        if args.list_models:
            assets = list_available_assets(args.target)
            for asset in assets:
                print(asset.reference)
            print(f"{len(assets)} exact assets; supported targets: s100, s100p, s600; no model loaded.")
            return 0
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires --target s100, s100p, or s600; no board detection performed.")
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if args.dry_run:
            print(json.dumps({
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "model_path_exists": selection.model_path.is_file(),
                "input": {"name": "input", "shape": [1, 3, 224, 224], "dtype": "float32"},
                "outputs": {name: {"shape": list(shape), "transform": "runtime metadata required: F32=raw_f32; integer+quant=dequant"} for name, shape in OUTPUT_SHAPES.items()},
                "source_manifest": selection.asset.source_path,
            }, indent=2, ensure_ascii=False))
            return 0
        from samples._shared.platforms import require_execution_target

        require_execution_target(selection.target)
        if not selection.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {selection.model_path}; prepare it explicitly with model/download.sh.")
        return _run(selection, args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
