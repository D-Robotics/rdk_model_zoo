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

"""Export MobileSAM image encoder ONNX for RDK-S quantization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


class MobileSAMImageEncoder(torch.nn.Module):
    """MobileSAM image encoder wrapper with normalization outside the model."""

    def __init__(self, checkpoint: str, image_size: int = 512):
        """Load the MobileSAM image encoder for fixed-size ONNX export.

        Args:
            checkpoint: Path to the upstream `mobile_sam.pt` checkpoint.
            image_size: Square image size used during ONNX export.
        """

        super().__init__()
        from ultralytics.models.sam.build import build_mobile_sam

        model = build_mobile_sam(checkpoint).eval()
        model.set_imgsz((image_size, image_size))
        self.image_encoder = model.image_encoder

    def forward(self, normalized_images):
        """Run the image encoder on pre-normalized image tensors.

        Args:
            normalized_images: NCHW float32 tensor named `normalized_images`.

        Returns:
            Image embedding tensor for the mask decoder.
        """

        return self.image_encoder(normalized_images)


def main() -> None:
    """Export the MobileSAM image encoder ONNX model."""

    parser = argparse.ArgumentParser(description="Export MobileSAM image encoder ONNX.")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/MobileSAM"), help="Path to cloned MobileSAM repository.")
    parser.add_argument("--weights", type=Path, default=Path("./workspace/MobileSAM/weights/mobile_sam.pt"), help="Path to mobile_sam.pt.")
    parser.add_argument("--output", type=Path, default=Path("./mobile_sam_image_encoder_norm_512_op11.onnx"), help="Output ONNX path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    model = MobileSAMImageEncoder(str(args.weights), args.size).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, args.size, args.size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["normalized_images"],
            output_names=["image_embeddings"],
            dynamic_axes=None,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
