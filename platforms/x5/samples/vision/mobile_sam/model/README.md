English | [简体中文](./README_cn.md)

# Model Files

This directory stores the RDK X5 runtime models for MobileSAM full-mask inference.

## Download

Run `bash download_model.sh` to download the `.bin` files from the RDK X5 Model Zoo archive, or regenerate them from the conversion flow. If the package is unavailable, follow `../conversion/README.md` to clone the official MobileSAM repository, export ONNX, and quantize with the matching YAML files.

- Official source repository: https://github.com/ChaoningZhang/MobileSAM
- Checkpoint source: official MobileSAM `weights/mobile_sam.pt`
- Encoder download URL: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_image_encoder_norm_512x512_allint16.bin
- Encoder conversion YAML: `../conversion/configs/mobile_sam_image_encoder_norm_512x512_config.yaml`
- Decoder download URL: https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/mobile_sam/mobile_sam_decoder_512_box_default.bin
- Decoder conversion YAML: `../conversion/configs/mobile_sam_decoder_512_box_default_config.yaml`

## Files

| File | Description |
| --- | --- |
| `mobile_sam_image_encoder_norm_512x512_allint16.bin` | Quantized TinyViT image encoder for `bayes-e` |
| `mobile_sam_decoder_512_box_default.bin` | Quantized box-prompt mask decoder for `bayes-e` |

## Interface

Encoder:

- Input: 
ormalized_images`, `1x3x512x512`, float32 NCHW
- Output: `image_embeddings`, `1x256x32x32`

Decoder:

- Inputs: `image_embeddings`, `1x256x32x32`; `boxes`, `1x4x1x1`
- Outputs: `low_res_masks`, `1x3x128x128`; `iou_predictions`, `1x3x1x1`
