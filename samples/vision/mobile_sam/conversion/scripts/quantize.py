"""Quantize MobileSAM encoder and decoder with OE `hb_mapper makertbin`.

Run this script inside the OpenExplorer Docker environment from the conversion
folder after ONNX export and calibration data preparation.
"""

from __future__ import annotations

import argparse
import subprocess

CONFIGS = [
    "configs/mobile_sam_image_encoder_norm_512x512_config.yaml",
    "configs/mobile_sam_decoder_512_box_default_config.yaml",
]


def main() -> None:
    """Run `hb_mapper makertbin` for one or both MobileSAM YAML files."""

    parser = argparse.ArgumentParser(description="Run hb_mapper makertbin for MobileSAM encoder and decoder.")
    parser.add_argument("--config", choices=CONFIGS, help="Quantization YAML to run. Omit to run both.", default=None)
    args = parser.parse_args()
    configs = [args.config] if args.config else CONFIGS
    for config in configs:
        cmd = ["hb_mapper", "makertbin", "--config", config, "--model-type", "onnx"]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()