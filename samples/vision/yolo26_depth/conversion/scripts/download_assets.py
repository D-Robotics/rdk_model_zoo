"""Download upstream YOLO26 Depth ONNX assets for RDK-S quantization.

Pure stdlib (``urllib``) so it runs inside the OpenExplorer Docker image without
extra Python dependencies. Fetches the pre-exported per-variant ONNX so the user
can skip the torch-based export step; to rebuild from the Ultralytics checkpoint
instead, see ``../export.py``.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE_URL = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s/yolo26_depth"
VARIANTS = ("n", "s", "m", "l", "x")


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
        name = f"yolo26{v}-depth-log.onnx"
        download(f"{BASE_URL}/{name}", args.out / name)
    print("[done] YOLO26 Depth ONNX assets ready")


if __name__ == "__main__":
    main()
