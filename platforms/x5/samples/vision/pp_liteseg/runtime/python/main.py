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

"""
PP-LiteSeg-STDC1 Inference Entry Script.

Demonstrates the standard BPU inference pipeline for PP-LiteSeg-STDC1
semantic segmentation on a single input image, following RDK Model Zoo
engineering standards.

Workflow:
    1) Parse CLI arguments for model, image, and parameters.
    2) Initialize PPLiteSegConfig and PPLiteSeg model wrapper.
    3) Execute full pipeline: pre_process -> forward -> post_process.
    4) Visualize and save the 3-panel result image (Original|Overlay|Seg).

Notes:
    - Requires hbm_runtime (RDK X5 firmware >= 3.5.0).
    - NV12 input conversion is handled inside PPLiteSeg.pre_process().
    - Output class map uses the Cityscapes 19-class palette.

Example:
    python3 main.py \\
        --model-path ../../model/pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin \\
        --test-img ../../test_data/street.jpg \\
        --output ../../test_data/result.jpg
"""

import argparse
import os
import sys

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../model"))
TEST_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../test_data"))

DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "pp_liteseg_stdc1_cityscapes_1024x512_nv12.bin")
DEFAULT_TEST_IMAGE = os.path.join(TEST_DATA_DIR, "street.jpg")
DEFAULT_OUTPUT_PATH = os.path.join(TEST_DATA_DIR, "result.jpg")

sys.path.insert(0, SCRIPT_DIR)
from pp_liteseg import PPLiteSegConfig, PPLiteSeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PP-LiteSeg-STDC1 Semantic Segmentation Inference")

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the BPU quantized *.bin model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE,
                        help="Path to the test input image.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path to save the output result image.")
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="Overlay blending alpha (0~1). Default: 0.55.")
    parser.add_argument("--input-width", type=int, default=1024,
                        help="Model input width. Default: 1024.")
    parser.add_argument("--input-height", type=int, default=512,
                        help="Model input height. Default: 512.")
    return parser.parse_args()


def main() -> None:
    """Run PP-LiteSeg-STDC1 inference on a single image."""

    args = parse_args()

    # 1. Initialize configuration and model wrapper
    config = PPLiteSegConfig(
        model_path=args.model_path,
        input_width=args.input_width,
        input_height=args.input_height,
        alpha=args.alpha,
    )
    model = PPLiteSeg(config)

    # 2. Print model info
    print(f"[Info] Model: {args.model_path}")
    mname = model._mname
    iname = model._iname
    oname = model._oname
    print(f"[Info] Input : {iname}  shape={model.model.input_shapes[mname][iname]}"
          f"  dtype={model.model.input_dtypes[mname][iname].name}")
    print(f"[Info] Output: {oname}  shape={model.model.output_shapes[mname][oname]}"
          f"  dtype={model.model.output_dtypes[mname][oname].name}")

    # 3. Load image
    print(f"[Info] Loading image: {args.test_img}")
    bgr = cv2.imread(args.test_img)
    if bgr is None:
        sys.exit(f"[Error] Cannot read image: {args.test_img}")

    # 4. Inference pipeline
    print("[Info] Running BPU inference...")
    seg = model.predict(bgr)

    # 5. Print detected classes
    import numpy as np
    from pp_liteseg import CITYSCAPES_CLASS_NAMES
    unique = sorted(np.unique(seg).tolist())
    valid = [c for c in unique if 0 <= c < len(CITYSCAPES_CLASS_NAMES)]
    print(f"[Info] Detected {len(valid)} classes: {[CITYSCAPES_CLASS_NAMES[c] for c in valid]}")

    # 6. Visualize and save
    result = model.visualize(bgr, seg)
    cv2.imwrite(args.output, result)
    print(f"[Info] Saved -> {args.output}  ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
