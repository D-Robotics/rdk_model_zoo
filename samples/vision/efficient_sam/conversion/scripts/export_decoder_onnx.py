"""Export EfficientSAM-Tiny fixed-prompt decoder ONNX for RDK X5 quantization."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch


class EfficientSAMFixedPromptDecoder(torch.nn.Module):
    """EfficientSAM decoder wrapper with fixed positive point prompts.

    The wrapper precomputes sparse prompt embeddings and dense positional
    encodings so the exported ONNX takes only image embeddings as input.
    """

    def __init__(self, model, image_size: int, points: list[float]):
        """Initialize prompt constants for fixed-prompt decoding.

        Args:
            model: Loaded EfficientSAM model that provides prompt and mask decoders.
            image_size: Export image size used to scale point coordinates.
            points: Four values describing two positive points as `x1 y1 x2 y2`.
        """

        super().__init__()
        self.mask_decoder = model.mask_decoder
        point_tensor = torch.tensor(
            [[[points[0], points[1]], [points[2], points[3]], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]],
            dtype=torch.float32,
        )
        labels = torch.tensor([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
        with torch.no_grad():
            point_tensor = torch.stack(
                [
                    point_tensor[..., 0] * model.image_encoder.img_size / image_size,
                    point_tensor[..., 1] * model.image_encoder.img_size / image_size,
                ],
                dim=-1,
            )
            sparse_embeddings = model.prompt_encoder(point_tensor, labels)
            image_pe = model.prompt_encoder.get_dense_pe()
        self.register_buffer("sparse_embeddings_const", sparse_embeddings)
        self.register_buffer("image_pe_const", image_pe)

    def forward(self, image_embeddings):
        """Decode masks and IoU predictions from image embeddings.

        Args:
            image_embeddings: Encoder output tensor with shape `1x256x32x32`.

        Returns:
            Tuple containing low-resolution mask logits and IoU predictions.
        """

        output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat((output_tokens, self.sparse_embeddings_const), dim=1)
        batch, channels, height, width = image_embeddings.shape
        hs, src = self.mask_decoder.transformer(image_embeddings, self.image_pe_const, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.mask_decoder.num_mask_tokens), :]
        upscaled_embedding = src.transpose(1, 2).view(batch, channels, height, width)
        for upscaling_layer in self.mask_decoder.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)
        hyper_in_list = []
        for index, output_hypernetworks_mlp in enumerate(self.mask_decoder.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, index, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        batch, channels, height, width = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(batch, channels, height * width)).view(batch, -1, height, width)
        iou_predictions = self.mask_decoder.iou_prediction_head(iou_token_out)
        return masks[:, 1:, :, :], iou_predictions[:, 1:]


def main() -> None:
    """Export the fixed-prompt decoder ONNX model."""

    parser = argparse.ArgumentParser(description="Export EfficientSAM-Tiny fixed-prompt decoder ONNX.")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/EfficientSAM"), help="Path to cloned EfficientSAM repository.")
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/EfficientSAM/weights/efficient_sam_vitt.pt"), help="Path to efficient_sam_vitt.pt.")
    parser.add_argument("--output", type=Path, default=Path("./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx"), help="Output ONNX path.")
    parser.add_argument("--size", type=int, default=512, help="Square image size.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version.")
    parser.add_argument("--points", nargs=4, type=float, default=[248.0, 210.0, 302.0, 315.0], help="Two positive points: x1 y1 x2 y2.")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

    old_cwd = os.getcwd()
    os.chdir(args.repo)
    try:
        model = build_efficient_sam_vitt().eval()
    finally:
        os.chdir(old_cwd)
    model.image_encoder.img_size = args.size
    model.image_encoder.image_embedding_size = args.size // 16
    model.prompt_encoder.input_image_size = (args.size, args.size)
    model.prompt_encoder.image_embedding_size = (args.size // 16, args.size // 16)
    decoder = EfficientSAMFixedPromptDecoder(model, args.size, args.points).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image_embeddings = torch.randn(1, 256, args.size // 16, args.size // 16, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (image_embeddings,),
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["image_embeddings"],
            output_names=["low_res_masks", "iou_predictions"],
            dynamic_axes=None,
        )
    print(f"Exported {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()