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

"""DINOv2 vision encoder inference entry script."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect

from dinov2 import CLS_FEAT, PATCH_FEAT, Dinov2, Dinov2Config

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DINOv2")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

MODEL_FILENAME = "dinov2_vits14_224_int16_{suffix}.hbm"

DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "dog.jpg")
DEFAULT_SECOND_IMAGE = os.path.join(TEST_DATA_DIR, "bus.jpg")


def default_model_path() -> str:
    """Resolve the default model path from the on-board SoC name."""

    board_type = ""
    try:
        with open("/sys/class/boardinfo/board_type") as fh:
            board_type = fh.read().strip()
    except OSError:
        pass
    _, march, suffix, _ = inspect.resolve_platform(
        inspect.get_soc_name_fallback_free(), board_type
    )
    return os.path.join(MODEL_DIR, march, MODEL_FILENAME.format(suffix=suffix))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="DINOv2 Vision Encoder")
    parser.add_argument("--model-path", type=str, default=default_model_path(), help="Path to the quantized DINOv2 .hbm model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to the first test image.")
    parser.add_argument("--second-img", type=str, default=DEFAULT_SECOND_IMAGE, help="Path to the second test image for the similarity demo.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input resolution expected by the DINOv2 model.")
    parser.add_argument("--output", type=str, default=CLS_FEAT, choices=[CLS_FEAT, PATCH_FEAT], help="Model output to inspect.")
    parser.add_argument("--priority", type=int, default=0, help="Runtime priority in the range 0 to 255.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes used by hbm_runtime.")
    return parser.parse_args()


def embedding_summary(tensor: "np.ndarray", name: str) -> dict:
    """Build a printable summary of one output tensor."""

    flat = tensor.reshape(-1).astype("float32")
    return {
        "output": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float((flat ** 2).sum() ** 0.5),
    }


def main() -> None:
    """Run DINOv2 inference on the test images and print output summaries."""

    args = parse_args()
    image = file_io.load_image(args.test_img)

    config = Dinov2Config(
        model_path=args.model_path,
        image_size=args.image_size,
        output=args.output,
    )
    model = Dinov2(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    inspect.print_model_info(model.model)

    feat_a = model.predict(image)
    logger.info("DINOv2 %s summary for %s:\n%s", args.output, args.test_img,
                json.dumps(embedding_summary(feat_a, args.output), indent=2, ensure_ascii=False))

    # Similarity demo: compare the first image against a second image with the
    # global embedding. Same-image cosine is 1.0; different images show how the
    # embedding separates content.
    if os.path.isfile(args.second_img):
        image_b = file_io.load_image(args.second_img)
        feat_b = model.predict(image_b)
        a = feat_a.reshape(-1).astype("float64")
        b = feat_b.reshape(-1).astype("float64")
        cosine = float(a @ b / ((a @ a) ** 0.5 * (b @ b) ** 0.5))
        logger.info("Cosine similarity (%s) between %s and %s: %.6f",
                    args.output, os.path.basename(args.test_img),
                    os.path.basename(args.second_img), cosine)


if __name__ == "__main__":
    main()
