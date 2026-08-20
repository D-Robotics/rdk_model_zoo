"""Shared data loading and numerical metrics for HIMLoco evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


INPUT_WIDTH = 270
OUTPUT_WIDTH = 12
ACTION_SCALE = 0.25


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rollout(
    path: Path,
    observation_key: str,
    action_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load policy inputs and optional recorded actions from a native rollout."""

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(artifact, torch.Tensor):
        observations = artifact
        actions = None
    elif isinstance(artifact, dict):
        if observation_key not in artifact:
            raise KeyError(f"{observation_key!r} is not present in {path}")
        observations = artifact[observation_key]
        actions = artifact.get(action_key) if action_key else None
    else:
        raise TypeError("rollout .pt must contain a Tensor or a dictionary of Tensors")

    observations = validate_tensor(observations, observation_key, INPUT_WIDTH)
    if actions is not None:
        actions = validate_tensor(actions, action_key or "actions", OUTPUT_WIDTH)
        if actions.shape[0] != observations.shape[0]:
            raise ValueError("observation/action sample counts do not match")
    return observations, actions


def validate_tensor(value: object, name: str, width: int) -> torch.Tensor:
    """Validate one finite rank-two tensor and normalize it to CPU float32."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name!r} must be a Tensor")
    if value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"expected {name} shape [N,{width}], got {tuple(value.shape)}")
    value = value.detach().cpu().to(torch.float32).contiguous()
    if value.shape[0] == 0:
        raise ValueError(f"{name!r} is empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name!r} contains NaN/Inf")
    return value


def select_indices(total: int, count: int, seed: int) -> np.ndarray:
    """Select sorted deterministic indices; zero means the complete rollout."""

    selected = total if count == 0 else count
    if selected < 1 or selected > total:
        raise ValueError(f"sample count must be in [1,{total}] or 0")
    if selected == total:
        return np.arange(total, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(total, size=selected, replace=False))


def cosine_similarity(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Calculate cosine similarity with deterministic zero-vector handling."""

    candidate64 = candidate.astype(np.float64).ravel()
    reference64 = reference.astype(np.float64).ravel()
    denominator = np.linalg.norm(candidate64) * np.linalg.norm(reference64)
    if denominator == 0.0:
        return 1.0 if np.array_equal(candidate64, reference64) else 0.0
    similarity = np.dot(candidate64, reference64) / denominator
    return float(np.clip(similarity, -1.0, 1.0))


def numerical_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int]:
    """Calculate action and joint-target error metrics."""

    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(
            f"shape mismatch: reference={reference.shape}, candidate={candidate.shape}"
        )
    if reference.shape[1] != OUTPUT_WIDTH:
        raise ValueError(
            f"expected action width {OUTPUT_WIDTH}, got {reference.shape[1]}"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("reference/candidate contains NaN/Inf")

    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    per_sample_cosine = [
        cosine_similarity(candidate[index], reference[index])
        for index in range(reference.shape[0])
    ]
    radians_to_degrees = 180.0 / np.pi
    return {
        "sample_count": int(reference.shape[0]),
        "action_mae": float(absolute.mean()),
        "action_rmse": float(np.sqrt(np.square(delta).mean())),
        "action_max_abs": float(absolute.max()),
        "global_cosine_similarity": cosine_similarity(candidate, reference),
        "minimum_sample_cosine_similarity": float(min(per_sample_cosine)),
        "joint_target_mae_rad": float(absolute.mean() * ACTION_SCALE),
        "joint_target_max_abs_rad": float(absolute.max() * ACTION_SCALE),
        "joint_target_mae_deg": float(
            absolute.mean() * ACTION_SCALE * radians_to_degrees
        ),
        "joint_target_max_abs_deg": float(
            absolute.max() * ACTION_SCALE * radians_to_degrees
        ),
    }


def write_report(path: Path, report: dict) -> None:
    """Write one new JSON report without replacing prior evidence."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
