"""Safe label-file parsing for the EfficientViT sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.labels`); this module keeps the EfficientViT import path.
"""

from __future__ import annotations

from samples._shared.labels import load_labels

__all__ = ["load_labels"]
