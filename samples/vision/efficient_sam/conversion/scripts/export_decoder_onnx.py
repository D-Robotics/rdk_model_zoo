# Copyright (c) 2025 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Export the EfficientSAM fixed two-point decoder for X5 or RDK-S."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


TARGETS = ("x5", "s100", "s100p", "s600")
DEFAULT_OUTPUTS = {
    "x5": "efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx",
    "s100": "efficient_sam_vitt_decoder_512_op11.onnx",
    "s100p": "efficient_sam_vitt_decoder_512_op11.onnx",
    "s600": "efficient_sam_vitt_decoder_512_op11.onnx",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export EfficientSAM fixed-prompt decoder ONNX.")
    parser.add_argument("--target", choices=TARGETS, default="x5")
    parser.add_argument("--repo", type=Path, default=Path("./workspace/EfficientSAM"))
    parser.add_argument("--checkpoint", type=Path, default=Path("./workspace/EfficientSAM/weights/efficient_sam_vitt.pt"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--points", nargs=4, type=float, default=[248.0, 210.0, 302.0, 315.0])
    return parser


def build_decoder(model, torch, image_size: int, points: list[float]):
    class FixedPromptDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mask_decoder = model.mask_decoder
            point_tensor = torch.tensor(
                [[[points[0], points[1]], [points[2], points[3]], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0]]],
                dtype=torch.float32,
            )
            labels = torch.tensor([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
            with torch.no_grad():
                point_tensor = torch.stack(
                    [point_tensor[..., 0] * model.image_encoder.img_size / image_size,
                     point_tensor[..., 1] * model.image_encoder.img_size / image_size], dim=-1)
                sparse_embeddings = model.prompt_encoder(point_tensor, labels)
                image_pe = model.prompt_encoder.get_dense_pe()
            self.register_buffer("sparse_embeddings_const", sparse_embeddings)
            self.register_buffer("image_pe_const", image_pe)

        def forward(self, image_embeddings):
            output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0).unsqueeze(0)
            tokens = torch.cat((output_tokens, self.sparse_embeddings_const), dim=1)
            batch, channels, height, width = image_embeddings.shape
            hs, src = self.mask_decoder.transformer(image_embeddings, self.image_pe_const, tokens)
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + self.mask_decoder.num_mask_tokens), :]
            upscaled_embedding = src.transpose(1, 2).view(batch, channels, height, width)
            for layer in self.mask_decoder.final_output_upscaling_layers:
                upscaled_embedding = layer(upscaled_embedding)
            hyper_in = torch.stack([mlp(mask_tokens_out[:, i, :]) for i, mlp in enumerate(self.mask_decoder.output_hypernetworks_mlps)], dim=1)
            batch, channels, height, width = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(batch, channels, height * width)).view(batch, -1, height, width)
            iou_predictions = self.mask_decoder.iou_prediction_head(iou_token_out)
            return masks[:, 1:, :, :], iou_predictions[:, 1:]

    return FixedPromptDecoder().eval()


def _load_model(args):
    if args.target == "x5":
        expected = (args.repo / "weights" / "efficient_sam_vitt.pt").resolve()
        supplied = args.checkpoint.resolve()
        if supplied != expected:
            raise ValueError(f"X5 builder uses only {expected}; custom --checkpoint is unsupported")
        if not expected.is_file():
            raise FileNotFoundError(expected)
    import torch

    sys.path.insert(0, str(args.repo))
    if args.target == "x5":
        from efficient_sam.build_efficient_sam import build_efficient_sam_vitt
        old_cwd = os.getcwd()
        os.chdir(args.repo)
        try:
            model = build_efficient_sam_vitt().eval()
        finally:
            os.chdir(old_cwd)
    else:
        from efficient_sam.efficient_sam import build_efficient_sam
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        model = build_efficient_sam(192, 3, checkpoint=str(args.checkpoint)).eval()
    return torch, model


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch, model = _load_model(args)
    model.image_encoder.img_size = args.size
    model.image_encoder.image_embedding_size = args.size // 16
    model.prompt_encoder.input_image_size = (args.size, args.size)
    model.prompt_encoder.image_embedding_size = (args.size // 16, args.size // 16)
    decoder = build_decoder(model, torch, args.size, args.points)
    output = args.output or Path(DEFAULT_OUTPUTS[args.target])
    output.parent.mkdir(parents=True, exist_ok=True)
    embedding = torch.randn(1, 256, args.size // 16, args.size // 16, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(decoder, (embedding,), str(output), export_params=True, opset_version=args.opset,
                          do_constant_folding=True, input_names=["image_embeddings"],
                          output_names=["low_res_masks", "iou_predictions"], dynamic_axes=None)
    print(f"Exported {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
