# Copyright (c) 2025 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantize MobileSAM encoder and decoder HBMs with OE ``hb_compile``.

Run this script inside the OpenExplorer Docker environment from the ``conversion``
folder, after ONNX export and calibration data preparation. One committed YAML is
provided per S-series march (``nash-e`` / ``nash-m`` / ``nash-p``); pick a march
or run all six configs.
"""

from __future__ import annotations

import argparse
import subprocess

MARCHES = ("nash-e", "nash-m", "nash-p")
CONFIGS = {
    "nash-e": (
        "configs/mobile_sam_encoder_nashe_config.yaml",
        "configs/mobile_sam_decoder_512_nashe_config.yaml",
    ),
    "nash-m": (
        "configs/mobile_sam_encoder_nashm_config.yaml",
        "configs/mobile_sam_decoder_512_nashm_config.yaml",
    ),
    "nash-p": (
        "configs/mobile_sam_encoder_nashp_config.yaml",
        "configs/mobile_sam_decoder_512_nashp_config.yaml",
    ),
}


def main() -> None:
    """Run ``hb_compile`` for one or all S-series marches."""

    parser = argparse.ArgumentParser(
        description="Run hb_compile for MobileSAM encoder+decoder HBMs."
    )
    parser.add_argument(
        "--march",
        choices=MARCHES,
        help="S-series march to compile. Omit to run all three marches.",
        default=None,
    )
    parser.add_argument(
        "--config",
        help="Specific config YAML path (overrides --march).",
        default=None,
    )
    args = parser.parse_args()

    if args.config:
        configs = [args.config]
    else:
        marches = [args.march] if args.march else list(MARCHES)
        configs = [cfg for march in marches for cfg in CONFIGS[march]]

    for config in configs:
        cmd = ["hb_compile", "--config", config]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
