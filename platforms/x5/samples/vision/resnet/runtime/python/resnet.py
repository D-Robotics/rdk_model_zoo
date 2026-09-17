"""Compatibility import for the historical X5 ResNet entrypoint.

The maintained implementation lives in the canonical ResNet18 sample.  This
module preserves the old import path and public class names so existing X5
applications continue to work while using the canonical contracts.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "samples/vision/resnet/runtime/python/legacy.py").is_file():
            return candidate
    raise RuntimeError("Could not locate the canonical samples/vision/resnet tree.")


_ROOT = _repository_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from samples.vision.resnet.runtime.python.legacy import (  # noqa: E402
    ResNet,
    ResNetConfig,
)


__all__ = ["ResNet", "ResNetConfig"]
