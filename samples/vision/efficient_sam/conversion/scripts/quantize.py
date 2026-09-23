# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Run the source quantizer for EfficientSAM without changing its recipe."""

from __future__ import annotations

import argparse
import subprocess

TARGETS = ("x5", "s100", "s100p", "s600")
CONFIGS = {
    "x5": ("configs/x5/efficient_sam_vitt_encoder_featuremap_config.yaml", "configs/x5/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml"),
    "s100": ("configs/s100/efficient_sam_encoder_nashe_config.yaml", "configs/s100/efficient_sam_decoder_nashe_config.yaml"),
    "s100p": ("configs/s100p/efficient_sam_encoder_nashm_config.yaml", "configs/s100p/efficient_sam_decoder_nashm_config.yaml"),
    "s600": ("configs/s600/efficient_sam_encoder_nashp_config.yaml", "configs/s600/efficient_sam_decoder_nashp_config.yaml"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize EfficientSAM encoder and decoder.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--config", default=None, help="Specific config path; overrides --target.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configs = [args.config] if args.config else list(CONFIGS[args.target])
    for config in configs:
        if args.target == "x5":
            command = ["hb_mapper", "makertbin", "--config", config, "--model-type", "onnx"]
        else:
            command = ["hb_compile", "--config", config]
        print("+", " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
