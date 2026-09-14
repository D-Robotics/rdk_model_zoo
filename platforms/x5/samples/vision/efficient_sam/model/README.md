English | [简体中文](./README_cn.md)

# Model Files

This directory stores the download script and documentation for the RDK X5 runtime models used by EfficientSAM-Tiny full-mask inference.

## Download

Run `bash download_model.sh` to download the `.bin` files from the RDK X5 Model Zoo archive, or regenerate them from the conversion flow. If the package is unavailable, follow `../conversion/README.md` to clone the official EfficientSAM repository, export ONNX, and quantize with the matching YAML files.

- Official source repository: https://github.com/yformer/EfficientSAM
- Checkpoint source: official EfficientSAM `weights/efficient_sam_vitt.pt`
- Encoder download URL: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficient_sam/efficient_sam_vitt_encoder_512x512_default_none.bin
- Encoder conversion YAML: `../conversion/configs/efficient_sam_vitt_encoder_featuremap_config.yaml`
- Decoder download URL: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/efficient_sam/efficient_sam_vitt_decoder_fixedprompt_512_default.bin
- Decoder conversion YAML: `../conversion/configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml`

## Files

| File | Description |
| --- | --- |
| `efficient_sam_vitt_encoder_512x512_default_none.bin` | Quantized EfficientSAM-Tiny image encoder for `bayes-e` |
| `efficient_sam_vitt_decoder_fixedprompt_512_default.bin` | Quantized fixed-prompt mask decoder for `bayes-e` |

## Interface

Encoder:

- Input: `batched_images`, `1x3x512x512`, float32 NCHW
- Output: `image_embeddings`, `1x256x32x32`

Decoder:

- Input: `image_embeddings`, `1x256x32x32`
- Outputs: `low_res_masks`, `1x3x128x128`; `iou_predictions`, `1x3x1x1`