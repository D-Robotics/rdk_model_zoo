# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""SDK-free CLI and board entrypoint for the R3D-18 sample."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_binding = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
CLASS_COUNT = _binding.CLASS_COUNT
INPUT_SHAPE = _binding.INPUT_SHAPE
SAMPLE_DIR = _binding.SAMPLE_DIR
SUPPORTED_TARGETS = _binding.SUPPORTED_TARGETS
list_available_assets = _binding.list_available_assets
resolve_selection = _binding.resolve_selection

DEFAULT_CLIP = SAMPLE_DIR / "test_data/video0.npy"
DEFAULT_LABELS = SAMPLE_DIR / "test_data/kinetics_classnames.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 3D ResNet-18 video action classification.")
    parser.add_argument("--target", choices=("auto", *SUPPORTED_TARGETS), default="auto")
    parser.add_argument("--asset-id", default=None, help="Exact manifest asset reference.")
    parser.add_argument("--model-path", default=None, help="External HBM path; requires --asset-id.")
    parser.add_argument("--test-clip", default=str(DEFAULT_CLIP), help="Preprocessed float32 clip .npy path.")
    parser.add_argument("--label-file", default=str(DEFAULT_LABELS), help="Kinetics-400 JSON mapping path.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions, 1..400.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority 0..255.")
    parser.add_argument("--bpu-cores", type=int, nargs="+", default=[0], help="Nonnegative BPU core indexes.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true", help="List exact manifest assets without loading SDK.")
    mode.add_argument("--dry-run", action="store_true", help="Resolve the contract without loading SDK or files.")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the stable CLI contract without reading board identity."""

    return build_parser().parse_args(argv)


def _report(result, labels: dict[int, str], selection, clip_path: Path) -> dict:
    return {
        "asset_id": selection.asset.reference,
        "target": selection.target,
        "clip": str(clip_path),
        "predictions": [
            {"class_id": int(class_id), "score": float(score), "label": labels.get(int(class_id), str(int(class_id)))}
            for class_id, score in zip(result.class_ids, result.scores)
        ],
    }


def _run(selection, args) -> int:
    import numpy as np

    task_module = importlib.import_module("samples.vision.3dresnet.runtime.python.classification")
    labels_module = importlib.import_module("samples.vision.3dresnet.runtime.python.labels")
    runner_module = importlib.import_module("samples.vision.3dresnet.runtime.python.model_runner")
    VideoClassificationTask = task_module.VideoClassificationTask
    load_labels = labels_module.load_labels
    RuntimeModelRunner = runner_module.RuntimeModelRunner

    clip_path = Path(args.test_clip).expanduser()
    clip = np.load(clip_path, allow_pickle=False)
    labels = load_labels(args.label_file)
    runner = RuntimeModelRunner(selection)
    binding = runner.load()
    runner.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    task = VideoClassificationTask(runner, binding, top_k=args.top_k, labels=labels)
    result = task.predict(clip)
    print(json.dumps(_report(result, labels, selection, clip_path), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.list_models:
            assets = list_available_assets(args.target)
            for asset in assets:
                print(asset.reference)
            print(f"{len(assets)} exact asset; no model loaded.")
            return 0
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires --target s100; no board detection performed.")
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        if args.dry_run:
            print(json.dumps({
                "target": selection.target,
                "asset_id": selection.asset.reference,
                "model_path": str(selection.model_path),
                "model_path_exists": selection.model_path.is_file(),
                "input": {"name": "runtime metadata input name", "shape": list(INPUT_SHAPE), "dtype": "float32"},
                "output": {"name": "runtime metadata output name", "shape": [1, CLASS_COUNT], "dtype": "float32", "semantic": "source logits"},
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
