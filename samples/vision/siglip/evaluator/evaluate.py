# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Run declared basic image relations on both encoder output roles."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.embedding_evaluator import (
    main as evaluate, evaluate_relation, evaluate_token_correspondence,
)


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
    from samples.vision.siglip.runtime.python.cli import resolve_selection
    from samples.vision.siglip.runtime.python.embedding import SigLIPEmbedder
    for role in ("pooler_output", "last_hidden_state"):
        selection = resolve_selection(args.target, asset_id=args.asset_id,
                                      model_path=args.model_path, submodel=role)
        yield role, SigLIPEmbedder(selection)

def evaluate_role(role, features):
    """Score global semantics or aligned patch input consistency.

    Args:
        role: Selected packed submodel output name.
        features: Arrays captured for the five declared input images.

    Returns:
        dict: Role-specific criterion results and diagnostic measurements.

    Raises:
        ValueError: Features or role violate the evaluation contract.
    """
    if role == "pooler_output":
        return dict(evaluate_relation(features), task="global image relation")
    if role == "last_hidden_state":
        return evaluate_token_correspondence(features)
    raise ValueError(f"Unknown SigLIP output role: {role}")


def main(argv=None):
    """Run the relation CLI and return its zero/pass or one/fail status."""
    return evaluate(build_models, argv, evaluate_role=evaluate_role, semantic_basis="pooler_output")


if __name__ == "__main__":
    raise SystemExit(main())
