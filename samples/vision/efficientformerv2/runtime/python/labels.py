"""Safe label-file parsing for the EfficientFormerV2 sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.labels`); this module keeps the EfficientFormerV2 import path.
"""

from __future__ import annotations

from samples._shared.labels import load_labels

__all__ = ["load_labels"]
