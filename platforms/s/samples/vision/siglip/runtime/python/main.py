# Copyright (c) 2026 D-Robotics Corporation
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

"""SigLIP vision encoder inference entry script."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect

from siglip import LAST_HIDDEN_STATE, POOLER_OUTPUT, SigLIP, SigLIPConfig


logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SigLIP")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "s100", "bpu-siglip-base-patch16-224.hbm")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "dog.jpg")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="SigLIP Vision Encoder")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the packed SigLIP .hbm model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the test image.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input resolution expected by the selected SigLIP model.")
    parser.add_argument("--submodel", type=str, default=POOLER_OUTPUT, choices=[POOLER_OUTPUT, LAST_HIDDEN_STATE], help="Packed HBM submodel to run.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes used by hbm_runtime.")
    return parser.parse_args()


def main() -> None:
    """Run SigLIP inference on one image and print output summary."""

    args = parse_args()
    image = file_io.load_image(args.test_img)
    config = SigLIPConfig(
        model_path=args.model_path,
        image_size=args.image_size,
        submodel=args.submodel,
    )
    model = SigLIP(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    result = model.predict(image)
    flat = result.reshape(-1).astype("float32")
    summary = {
        "submodel": args.submodel,
        "shape": list(result.shape),
        "dtype": str(result.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float((flat ** 2).sum() ** 0.5),
    }
    logger.info("SigLIP output summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
