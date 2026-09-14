#!/usr/bin/env python3
"""Compare decoded S100P/S600 outputs with DiffusionDrive float references."""

from __future__ import annotations

import argparse

import numpy as np


def cosine(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Compute flattened cosine similarity for two tensors.

    Args:
        lhs: First tensor.
        rhs: Second tensor with the same number of elements.

    Returns:
        Flattened cosine similarity.
    """

    lhs64 = lhs.astype(np.float64).reshape(-1)
    rhs64 = rhs.astype(np.float64).reshape(-1)
    return float(np.dot(lhs64, rhs64) / (np.linalg.norm(lhs64) * np.linalg.norm(rhs64) + 1e-12))


def class_distribution(labels: np.ndarray) -> dict[int, float]:
    """Calculate the pixel proportion of each present BEV class.

    Args:
        labels: Integer BEV class map.

    Returns:
        Mapping from class ID to pixel proportion.
    """

    values, counts = np.unique(labels, return_counts=True)
    total = float(labels.size)
    return {int(value): float(count / total) for value, count in zip(values, counts)}


def semantic_iou(reference: np.ndarray, board: np.ndarray, num_classes: int = 7) -> dict[int, float]:
    """Calculate per-class intersection over union for BEV semantics.

    Args:
        reference: Float-model BEV class map.
        board: Board-model BEV class map.
        num_classes: Total number of semantic classes.

    Returns:
        Mapping from class ID to IoU for classes present in either map.
    """

    metrics = {}
    for class_id in range(num_classes):
        reference_mask = reference == class_id
        board_mask = board == class_id
        union = np.logical_or(reference_mask, board_mask).sum()
        if union:
            metrics[class_id] = float(np.logical_and(reference_mask, board_mask).sum() / union)
    return metrics


def main() -> None:
    """Print output accuracy metrics for one reference sample.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Compare DiffusionDrive board and float outputs")
    parser.add_argument("--reference-npz", type=str, required=True, help="Path to float reference output NPZ.")
    parser.add_argument("--board-npz", type=str, required=True, help="Path to decoded board output NPZ.")
    args = parser.parse_args()
    with np.load(args.reference_npz, allow_pickle=False) as reference, np.load(args.board_npz, allow_pickle=False) as board:
        mapping = {"trajectory": "trajectory", "agent_states": "agent_states", "agent_labels": "agent_scores", "bev_semantic_map": "bev_logits"}
        for ref_name, board_name in mapping.items():
            lhs = np.asarray(reference[ref_name], dtype=np.float32)
            rhs = np.asarray(board[board_name], dtype=np.float32)
            if ref_name == "agent_labels":
                lhs = 1.0 / (1.0 + np.exp(-np.clip(lhs, -60.0, 60.0)))
            print(f"{ref_name}: cosine={cosine(lhs, rhs):.6f}, mae={np.mean(np.abs(lhs-rhs)):.6f}")
        ref_labels = np.argmax(reference["bev_semantic_map"], axis=1)
        board_labels = board["bev_labels"]
        agreement = np.mean(ref_labels == board_labels)
        print(f"bev_argmax_pixel_agreement={agreement:.6f}")
        print("bev_reference_distribution=", class_distribution(ref_labels))
        print("bev_board_distribution=", class_distribution(board_labels))
        iou = semantic_iou(ref_labels, board_labels)
        print("bev_class_iou=", iou)
        print(f"bev_mean_iou={np.mean(list(iou.values())):.6f}")


if __name__ == "__main__":
    main()
