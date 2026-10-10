# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run declared basic image relations on both encoder output roles."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.embedding_evaluator import main as evaluate


def build_models(args):
    """Yield each output role and its loaded sample model.

    Args:
        args: Evaluator selection arguments, including target and model path.

    Yields:
        tuple: Output role name and model exposing predict(image).

    Raises:
        ValueError: Selection or board metadata differs from the contract.
        RuntimeError: Board SDK loading fails.
    """
    from samples.vision.dinov2.runtime.python.cli import resolve_selection
    from samples.vision.dinov2.runtime.python.embedding import DINOv2Embedder
    selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
    cls_model = DINOv2Embedder(selection, output="cls_feat")
    yield "cls_feat", cls_model
    yield "patch_mean", DINOv2Embedder(selection, output="patch_feat", runner=cls_model.runner)

def main(argv=None):
    """Run the relation CLI and return its zero/pass or one/fail status."""
    return evaluate(build_models, argv)


if __name__ == "__main__":
    raise SystemExit(main())
