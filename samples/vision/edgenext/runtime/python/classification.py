"""Reusable single-image classification flow for the EdgeNeXt sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.classification`); this module is the EdgeNeXt import
path for that surface.
"""

from __future__ import annotations

from samples._shared.classification import (  # noqa: F401 - re-exported surface
    ClassificationResult,
    ClassificationTask,
    topk_from_logits,
    topk_from_scores,
)

__all__ = [
    "ClassificationResult",
    "ClassificationTask",
    "topk_from_logits",
    "topk_from_scores",
]
