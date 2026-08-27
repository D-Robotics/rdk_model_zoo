"""Download pre-exported YOLO26 Depth ONNX assets for RDK-S quantization.

Pure stdlib (``urllib``) so it runs inside the OpenExplorer Docker image without
extra Python dependencies. The filename follows the mixed release profile:

- ``n`` / ``s`` / ``m`` -> ``yolo26{v}-depth-log.onnx`` (NV12 boundary,
  calibrated log-depth in-graph)
- ``l`` / ``x`` -> ``yolo26{v}-depth_op11_lite.onnx`` (lite boundary, raw
  192x192 depth logit)

To rebuild any ONNX from the Ultralytics checkpoint instead, see
``../export.py`` (``--boundary`` follows the same variant mapping).

The ONNX assets are march-independent (one copy serves all three marches) and
live under the ``rdk_s100`` server tree: ``onnx/`` beside the per-march model
directories.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/yolo26_depth/onnx"
VARIANTS = ("n", "s", "m", "l", "x")
LITE_VARIANTS = ("l", "x")


def asset_name(variant: str) -> str:
    """Return the pre-exported ONNX filename for one variant."""
    if variant in LITE_VARIANTS:
        return f"yolo26{variant}-depth_op11_lite.onnx"
    return f"yolo26{variant}-depth-log.onnx"


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` if missing."""
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[get] {url} -> {dest}")
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        handle.write(response.read())


def main() -> None:
    """Fetch the per-variant YOLO26 Depth ONNX assets."""
    parser = argparse.ArgumentParser(description="Download YOLO26 Depth ONNX assets.")
    parser.add_argument("--out", type=Path, default=Path(".."), help="Output directory (conversion folder).")
    parser.add_argument("--variant", choices=VARIANTS, default=None, help="One variant; omit for all five.")
    args = parser.parse_args()

    variants = [args.variant] if args.variant else list(VARIANTS)
    for v in variants:
        name = asset_name(v)
        download(f"{BASE_URL}/{name}", args.out / name)
    print("[done] YOLO26 Depth ONNX assets ready")


if __name__ == "__main__":
    main()
