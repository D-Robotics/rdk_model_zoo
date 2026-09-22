"""Safe label-file parsing for the MobileNetV4 sample.

The implementation is shared by the unified classification samples
(:mod:`samples._shared.labels`); this module keeps the MobileNetV4 import path.
"""

from __future__ import annotations

from samples._shared.labels import load_labels

__all__ = ["load_labels"]
