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

"""Dump one EfficientSAM encoder embedding for decoder calibration.

The decoder is calibrated from real encoder embeddings. Run the exported float
encoder ONNX on a single image and write the ``image_embeddings`` output as a
raw float32 ``.bin`` file, which
``prepare_efficient_decoder_calibration.py --embedding`` then consumes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> None:
    """Run the encoder ONNX on one image and dump its embedding as a raw bin."""

    parser = argparse.ArgumentParser(description="Dump one EfficientSAM encoder embedding.")
    parser.add_argument("--onnx", type=Path, default=Path("./efficient_sam_vitt_encoder_512_op11.onnx"), help="Encoder ONNX path.")
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--output", type=Path, default=Path("./encoder_embedding.bin"), help="Output raw embedding path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
    tensor = np.transpose(image, (2, 0, 1))[None].astype(np.float32) / 255.0

    session = ort.InferenceSession(str(args.onnx))
    embedding = session.run(["image_embeddings"], {"batched_images": tensor})[0]
    embedding.astype(np.float32).tofile(str(args.output))
    print(f"Wrote {args.output} (shape {embedding.shape}, dtype float32)")


if __name__ == "__main__":
    main()