"""Quantize YOLO26 Depth HBMs with OE ``hb_compile``.

Run inside the OpenExplorer Docker image from this ``conversion/`` folder, after
ONNX export and calibration data preparation. One committed YAML per
(variant, march) is baked under ``ptq_yamls/``. The mixed release profile
selects the compile per variant:

- ``n`` / ``s`` / ``m``: NV12 profile (calibrated log-depth ONNX, letterboxed
  NV12 input, ``max`` calibration with percentile ``0.9999``).
- ``l`` / ``x``: lite profile (raw-logit ONNX, float32 featuremap input,
  ``default`` KL calibration).
"""

from __future__ import annotations

import argparse
import subprocess

VARIANTS = ("n", "s", "m", "l", "x")
MARCHES = ("nash-e", "nash-m", "nash-p")
LITE_VARIANTS = ("l", "x")


def config_path(variant: str, march: str) -> str:
    """Return the committed YAML path for one (variant, march) pair."""
    suffix = march.replace("-", "")
    if variant in LITE_VARIANTS:
        return f"ptq_yamls/yolo26{variant}_depth_lite_{suffix}_768.yaml"
    return f"ptq_yamls/yolo26{variant}_depth_{suffix}_768x768_nv12.yaml"


def main() -> None:
    """Run ``hb_compile`` for the selected variants and marches."""

    parser = argparse.ArgumentParser(description="Run hb_compile for YOLO26 Depth HBMs.")
    parser.add_argument("--variant", choices=VARIANTS, default=None, help="Model variant; omit for all.")
    parser.add_argument("--march", choices=MARCHES, default=None, help="S-series march; omit for all.")
    parser.add_argument("--config", default=None, help="Specific config YAML path (overrides --variant/--march).")
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    else:
        variants = [args.variant] if args.variant else list(VARIANTS)
        marches = [args.march] if args.march else list(MARCHES)
        configs = [config_path(v, m) for v in variants for m in marches]

    for config in configs:
        cmd = ["hb_compile", "--config", config]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
