English | [简体中文](./README_cn.md)

# MobileSAM Model Description

This sample provides an RDK X5 demo for MobileSAM full-mask segmentation. Both the TinyViT image encoder and box-prompt mask decoder are quantized to `.bin` models and run on board through `hbm_runtime`.

## Source Repository

- Official repository: https://github.com/ChaoningZhang/MobileSAM
- Checkpoint: `weights/mobile_sam.pt` from the official MobileSAM release
- Export input size: `512x512`

## Algorithm Overview

- **Task**: promptable image segmentation
- **Prompt**: box prompt `[185, 120, 380, 445]`
- **Encoder input**: `normalized_images`, `1x3x512x512`, NCHW float32 featuremap
- **Encoder output**: `image_embeddings`, `1x256x32x32`
- **Decoder input**: `image_embeddings` and `boxes`
- **Demo output**: full-size `512x512` binary mask and overlay image

## Algorithm Capabilities

- Runs a single-image MobileSAM segmentation demo on RDK X5.
- Uses one fixed-size `512x512` input image and one box prompt in resized image coordinates.
- Executes the TinyViT image encoder and box-prompt mask decoder as two quantized `.bin` models.
- Saves both a binary mask and a mask-overlay visualization.

## Algorithm Features

- The default prompt is the fixed box `[185, 120, 380, 445]`; it can be changed with the runtime `--box` argument.
- The sample focuses on board-side encoder + decoder inference with `hbm_runtime`.
- Interactive multi-prompt UI, point prompts, and batch image processing are not included in this demo.

## Directory Structure

```text
.
|-- conversion
|   |-- configs
|   |   |-- mobile_sam_image_encoder_norm_512x512_config.yaml
|   |   `-- mobile_sam_decoder_512_box_default_config.yaml
|   |-- scripts
|   |   |-- download_assets.py
|   |   |-- export_encoder_onnx.py
|   |   |-- export_decoder_onnx.py
|   |   |-- prepare_calibration.py
|   |   |-- prepare_decoder_calibration.py
|   |   `-- quantize.py
|   `-- README.md
|-- evaluator
|   |-- README.md
|   -- README_cn.md
|-- model
|   |-- download_model.sh
|   `-- README.md
|-- runtime
|   `-- python
|       |-- run.sh
|       |-- main.py
|       |-- mobile_sam.py
|       `-- README.md
|-- test_data
|   |-- dogs.jpg
|   |-- mobile_sam_binary_mask.png
|   `-- mobile_sam_full_mask_result.jpg
`-- README.md
```

## Quick Start

Run on an RDK X5 board with `hbm_runtime`:

```bash
cd samples/vision/mobile_sam/runtime/python
bash run.sh
```

Outputs:

- `test_data/mobile_sam_full_mask_result.jpg`
- `test_data/mobile_sam_binary_mask.png`

## Conversion Summary

1. Clone/download the official MobileSAM repository with `conversion/scripts/download_assets.py`.
2. Export `mobile_sam_image_encoder_norm_512_op11.onnx` with `conversion/scripts/export_encoder_onnx.py`.
3. Export `mobile_sam_decoder_512_box_op11.onnx` with `conversion/scripts/export_decoder_onnx.py`.
4. Prepare encoder calibration images with `conversion/scripts/prepare_calibration.py`.
5. Prepare decoder featuremap/box calibration data with `conversion/scripts/prepare_decoder_calibration.py`.
6. Quantize using:
   - `conversion/configs/mobile_sam_image_encoder_norm_512x512_config.yaml`
   - `conversion/configs/mobile_sam_decoder_512_box_default_config.yaml`

See `conversion/README.md` for exact commands.

## Model Evaluation

See `evaluator/README.md` for multi-thread `hrt_model_exec perf` performance results.

## Validation

- Encoder final cosine: `0.961277`.
- Decoder cosine: `low_res_masks=0.997539`, `iou_predictions=0.999972`.
- Board demo verified with `hbm_runtime` dual `.bin` inference.

## License

This sample follows the license terms of the upstream MobileSAM project and the RDK Model Zoo repository. Check the official MobileSAM repository for third-party model and checkpoint usage requirements.
