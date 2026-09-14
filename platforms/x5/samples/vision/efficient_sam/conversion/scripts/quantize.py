"""Quantize EfficientSAM-Tiny encoder and fixed-prompt decoder.

Run inside the OpenExplorer Docker environment from the conversion folder after
ONNX export and calibration data preparation.
"""

from __future__ import annotations

import argparse
import subprocess

CONFIGS = [
    "configs/efficient_sam_vitt_encoder_featuremap_config.yaml",
    "configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml",
]


def main() -> None:
    """Run `hb_mapper makertbin` for one or both EfficientSAM YAML files."""

    parser = argparse.ArgumentParser(description="Run hb_mapper makertbin for EfficientSAM encoder and decoder.")
    parser.add_argument("--config", choices=CONFIGS, help="Quantization YAML to run. Omit to run both.", default=None)
    args = parser.parse_args()
    configs = [args.config] if args.config else CONFIGS
    for config in configs:
        cmd = ["hb_mapper", "makertbin", "--config", config, "--model-type", "onnx"]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()