English | [简体中文](./README_cn.md)

# EfficientSAM-Tiny Model Description

This sample provides an RDK X5 demo for EfficientSAM-Tiny full-mask segmentation. Both the image encoder and fixed-prompt mask decoder are quantized to `.bin` models and run on board through `hbm_runtime`.

## Source Repository

- Official repository: https://github.com/yformer/EfficientSAM
- Checkpoint: `weights/efficient_sam_vitt.pt` from the official EfficientSAM release
- Export input size: `512x512`

## Algorithm Overview

- **Task**: promptable image segmentation
- **Prompt**: fixed positive points baked into decoder ONNX: `(248,210)` and `(302,315)`
- **Encoder input**: `batched_images`, `1x3x512x512`, NCHW float32 featuremap
- **Encoder output**: `image_embeddings`, `1x256x32x32`
- **Decoder input**: `image_embeddings`
- **Demo output**: full-size `512x512` binary mask and overlay image

## Algorithm Capabilities

- Runs a single-image EfficientSAM-Tiny segmentation demo on RDK X5.
- Uses one fixed-size `512x512` input image and two fixed positive point prompts baked into the decoder model.
- Executes the image encoder and fixed-prompt decoder as two quantized `.bin` models.
- Saves both a binary mask and a mask-overlay visualization.

## Algorithm Features

- The default prompts are fixed points `(248,210)` and `(302,315)` in resized image coordinates.
- The sample focuses on board-side encoder + decoder inference with `hbm_runtime`.
- Interactive multi-prompt UI, runtime point selection, and batch image processing are not included in this demo.

## Directory Structure

```text
.
|-- conversion
|   |-- configs
|   |   |-- efficient_sam_vitt_encoder_featuremap_config.yaml
|   |   `-- efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
|   |-- scripts
|   |   |-- download_assets.py
|   |   |-- export_encoder_onnx.py
|   |   |-- export_decoder_onnx.py
|   |   |-- prepare_calibration.py
|   |   |-- prepare_efficient_decoder_calibration.py
|   |   `-- quantize.py
|   |-- README.md
|   |-- QUANTIZATION_STATUS.md
|   `-- VALIDATION.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download_model.sh
|   `-- README.md
|-- runtime
|   `-- python
|       |-- run.sh
|       |-- main.py
|       |-- efficient_sam.py
|       `-- README.md
|-- test_data
|   |-- dogs.jpg
|   |-- efficient_sam_binary_mask.png
|   `-- efficient_sam_full_mask_result.jpg
`-- README.md
```

## Quick Start

Run on an RDK X5 board with `hbm_runtime`:

```bash
cd samples/vision/efficient_sam/runtime/python
bash run.sh
```

Outputs:

- `test_data/efficient_sam_full_mask_result.jpg`
- `test_data/efficient_sam_binary_mask.png`

## Conversion Summary

1. Clone/download the official EfficientSAM repository with `conversion/scripts/download_assets.py`.
2. Export `efficient_sam_vitt_encoder_512_splitqkv_op11.onnx` with `conversion/scripts/export_encoder_onnx.py`.
3. Export `efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx` with `conversion/scripts/export_decoder_onnx.py`.
4. Prepare encoder calibration images with `conversion/scripts/prepare_calibration.py`.
5. Prepare decoder embedding calibration data with `conversion/scripts/prepare_efficient_decoder_calibration.py`.
6. Quantize using:
   - `conversion/configs/efficient_sam_vitt_encoder_featuremap_config.yaml`
   - `conversion/configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml`

Generated ONNX files, `*_quant_info.json`, calibration tensors, and quantized `.bin` outputs are produced by the conversion flow and are not committed with the sample. See `conversion/README.md` for exact commands.

## Model Evaluation

No dataset-level evaluator is included. See `evaluator/README.md` for `hrt_model_exec perf` performance validation and `conversion/VALIDATION.md` for quantization evidence.

## Validation

- Encoder final cosine: `0.968013`.
- Fixed-prompt decoder cosine: `low_res_masks=0.965641`, `iou_predictions=0.997313`.
- Board demo verified with `hbm_runtime` dual `.bin` inference.

## License

This sample follows the license terms of the upstream EfficientSAM project and the RDK Model Zoo repository. Check the official EfficientSAM repository for third-party model and checkpoint usage requirements.