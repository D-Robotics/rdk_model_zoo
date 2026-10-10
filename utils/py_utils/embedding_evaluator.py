# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check a small declared image relation on the actual compiled encoder.

This is a basic semantic smoke check, not dataset accuracy. The independent
positive and negative must be chosen from their image content before running.
Token outputs are averaged over the patch axis; pooler/CLS use their vector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np



def image_vector(value):
    """Reduce one floating embedding to its normalized global vector.

    Args:
        value: Floating features shaped [1,D] or [1,tokens,D].

    Returns:
        np.ndarray: Owned float64 unit vector; tokens are averaged first.

    Raises:
        ValueError: Shape, dtype, finiteness, or norm is invalid.
    """
    value = np.asarray(value)
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError("Semantic comparison requires finite floating features; dequantize integer outputs first")
    if value.ndim not in (2, 3) or value.shape[0] != 1:
        raise ValueError("Expected [1,D] or [1,tokens,D] embedding")
    vector = value[0].astype(np.float64)
    if vector.ndim == 2:
        vector = vector.mean(axis=0)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite nonzero norm")
    return vector / norm


def evaluate_relation(features):
    """Score preset stability and positive-vs-negative cosine criteria.

    Args:
        features: Anchor, repeat, mild, positive, and negative feature arrays.

    Returns:
        dict: Cosines, declared criteria, individual checks, and pass status.

    Raises:
        ValueError: Feature names, dimensions, or numeric values are invalid.
    """
    expected = {"anchor", "repeat", "mild", "positive", "negative"}
    if set(features) != expected:
        raise ValueError("Expected anchor/repeat/mild/positive/negative features")
    vectors = {key: image_vector(value) for key, value in features.items()}
    if len({value.shape for value in vectors.values()}) != 1:
        raise ValueError("All compared feature dimensions must match")
    cosines = {key: float(np.clip(vectors["anchor"] @ vectors[key], -1, 1))
               for key in ("repeat", "mild", "positive", "negative")}
    criteria = {"repeat_min": 0.99999, "mild_min": 0.95, "semantic_margin_min": 0.05}
    margin = cosines["positive"] - cosines["negative"]
    checks = {"repeat": cosines["repeat"] >= criteria["repeat_min"],
              "mild": cosines["mild"] >= criteria["mild_min"],
              "semantic_relation": margin > criteria["semantic_margin_min"]}
    return {"cosines": cosines, "semantic_margin": margin, "criteria": criteria,
            "checks": checks, "passed": all(checks.values())}


def evaluate_token_correspondence(features):
    """Check aligned patch responses for repeat and mildly transformed inputs.

    Args:
        features: Anchor, repeat, mild, positive, and negative [1,tokens,D]
            arrays. Positive is retained for the mean-vector diagnostic only.

    Returns:
        dict: Gating input-consistency checks and the original, nongating
        mean-vector image-relation diagnostic.

    Raises:
        ValueError: Feature shape, dtype, norm, or dimensions are invalid.
    """
    diagnostic = evaluate_relation(features)
    arrays = {key: np.asarray(value) for key, value in features.items()}
    if any(value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 2
           for value in arrays.values()) or len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("Token correspondence requires equally shaped [1,tokens,D] arrays")
    # Preserve patch positions. Averaging loses the spatial feature response
    # and is not equivalent to SigLIP's learned attention pooling head.
    vectors = {key: image_vector(value.reshape(1, -1)) for key, value in arrays.items()}
    cosines = {key: float(np.clip(vectors["anchor"] @ vectors[key], -1, 1))
               for key in ("repeat", "mild", "negative")}
    criteria = {"repeat_min": 0.99999, "mild_min": 0.95,
                "aligned_mild_minus_negative_min": 0.05}
    gap = cosines["mild"] - cosines["negative"]
    checks = {"repeat": cosines["repeat"] >= criteria["repeat_min"],
              "mild": cosines["mild"] >= criteria["mild_min"],
              "input_correspondence": gap > criteria["aligned_mild_minus_negative_min"]}
    return {"task": "aligned token input consistency",
            "semantic_image_retrieval": False, "cosines": cosines,
            "aligned_mild_minus_negative": gap, "criteria": criteria, "checks": checks,
            "passed": all(checks.values()),
            "mean_image_relation_diagnostic": dict(diagnostic, affects_pass_criterion=False)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(build_models, argv=None, *, evaluate_role=None, semantic_basis=None):
    """Run declared image relations and save model-bound numeric evidence.

    Args:
        build_models: Callable yielding (role, loaded model) pairs for arguments.
        argv: Optional CLI arguments; None reads sys.argv.
        evaluate_role: Optional callable scoring (role, features); defaults to
            the global image-relation scorer.
        semantic_basis: Optional declared output role supporting image semantics.

    Returns:
        int: Zero if all roles pass, one for recorded model/numeric failures.

    Raises:
        ValueError: Input images are unreadable or positive duplicates anchor.
        OSError: Input or report files cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {key: getattr(args, key).resolve() for key in ("anchor", "positive", "negative")}
    if digest(paths["positive"]) == digest(paths["anchor"]):
        raise ValueError("Positive must be an independently different image")
    images = {key: cv2.imread(str(path)) for key, path in paths.items()}
    if any(value is None for value in images.values()):
        raise ValueError("Cannot read a declared input image")
    images["repeat"] = images["anchor"].copy()
    images["mild"] = np.clip(images["anchor"].astype(np.float32) * 0.9 + 5, 0, 255).astype(np.uint8)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rdk-model-zoo/embedding-relations/v1", "target": args.target,
              "asset_id": args.asset_id, "model_sha256": digest(args.model_path),
              "inputs": {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()},
              "mild_transform": "clip(BGR * 0.9 + 5, 0, 255).astype(uint8)",
              "scope": "basic independently declared image relation; not dataset accuracy", "roles": {}}
    if semantic_basis is not None:
        report["artifact_semantic_basis"] = semantic_basis
    try:
        for role, model in build_models(args):
            role_report = {}
            report["roles"][role] = role_report
            features = {}
            try:
                from utils.py_utils.runtime_meta import RuntimeMetadata, metadata_evidence
                runner = model.runner
                metadata = getattr(runner, "metadata", None)
                if metadata is None:
                    metadata = RuntimeMetadata.from_runtime(runner._runtime, model_name=model.binding.model_name)
                role_report["metadata"] = metadata_evidence(metadata)
                for key, image in images.items():
                    features[key] = np.array(model.predict(image), copy=True)
                role_report.update(evaluate_role(role, features) if evaluate_role is not None else evaluate_relation(features))
                role_report["feature_shape"] = list(features["anchor"].shape)
                role_report["feature_dtype"] = str(features["anchor"].dtype)
            except Exception as error:
                role_report.update(passed=False, error_type=type(error).__name__, error=str(error))
            finally:
                if features:
                    np.savez(args.output_dir / (role + ".npz"), **features)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["passed"] = bool(report["roles"]) and "error" not in report and all(value["passed"] for value in report["roles"].values())
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1
