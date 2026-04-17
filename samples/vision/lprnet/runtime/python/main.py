"""
LPRNet recognition inference entry script.

This module provides the standardized Python entry for the LPRNet sample on
RDK X5. The script is responsible for parsing command-line arguments,
constructing the runtime wrapper, running inference on the bundled binary
input tensor, and printing the decoded license plate string.
"""

from __future__ import annotations

import argparse
import logging
import os

from lprnet import LPRNet, LPRNetConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LPRNet")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "lpr.bin")
DEFAULT_TEST_BIN = os.path.join(TEST_DATA_DIR, "test.bin")


def main() -> None:
    """
    Run the complete LPRNet recognition pipeline on one binary input tensor.

    The entry follows the standardized sample pattern used in this repository:
    1. Parse default-usable command-line arguments.
    2. Build the LPRNet runtime configuration.
    3. Configure runtime scheduling parameters.
    4. Execute `predict()` on the runtime wrapper.
    5. Print the final decoded plate string.
    """

    parser = argparse.ArgumentParser(description="LPRNet Recognition Inference")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--priority", type=int, default=5, help="Model priority (0~255).")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes to run inference.")
    parser.add_argument("--test-bin", type=str, default=DEFAULT_TEST_BIN, help="Path to the packed float32 input tensor.")
    args = parser.parse_args()

    config = LPRNetConfig(model_path=args.model_path, test_bin=args.test_bin)
    model = LPRNet(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    plate = model.predict()
    logger.info(f"Recognized plate: {plate}")


if __name__ == "__main__":
    main()
