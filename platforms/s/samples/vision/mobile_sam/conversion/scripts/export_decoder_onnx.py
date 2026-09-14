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

"""Export MobileSAM box-prompt decoder ONNX for RDK-S quantization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


class MobileSAMDecoder(torch.nn.Module):
    """MobileSAM decoder wrapper that keeps the box prompt as ONNX input."""

    def __init__(self, checkpoint: str, image_size: int):
        """Load MobileSAM prompt encoder and mask decoder.

        Args:
            checkpoint: Path to the upstream `mobile_sam.pt` checkpoint.
            image_size: Square image size used during ONNX export.
        """

        super().__init__()
        from ultralytics.models.sam.build import build_mobile_sam

        model = build_mobile_sam(checkpoint).eval()
        model.set_imgsz((image_size, image_size))
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder

    def forward(self, image_embeddings, boxes):
        """Decode low-resolution masks from image embeddings and boxes.

        Args:
            image_embeddings: Encoder output tensor with shape `1x256x32x32`.
            boxes: Box prompt tensor in resized image coordinates.

        Returns:
            Tuple containing low-resolution masks and IoU predictions.
        """

        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=boxes, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        return low_res_masks, iou_predictions


def main() -> None:
    """Export the MobileSAM box-prompt decoder ONNX model."""

    parser = argparse.ArgumentParser(description="Export MobileSAM decoder ONNX.")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/MobileSAM"), help="Path to cloned MobileSAM repository.")
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/MobileSAM/weights/mobile_sam.pt"), help="Path to mobile_sam.pt.")
    parser.add_argument("--output", type=Path, default=Path("./mobile_sam_decoder_512_op11.onnx"), help="Output ONNX path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--box", nargs=4, type=float, default=[185.0, 120.0, 380.0, 445.0], help="Example export box x1 y1 x2 y2.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = MobileSAMDecoder(str(args.checkpoint), args.size).eval()
    image_embeddings = torch.randn(1, 256, args.size // 16, args.size // 16, dtype=torch.float32)
    boxes = torch.tensor([args.box], dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (image_embeddings, boxes),
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["image_embeddings", "boxes"],
            output_names=["low_res_masks", "iou_predictions"],
            dynamic_axes=None,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
