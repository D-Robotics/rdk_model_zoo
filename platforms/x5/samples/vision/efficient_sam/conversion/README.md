English | [简体中文](./README_cn.md)

# EfficientSAM-Tiny Conversion Guide

This directory documents the complete EfficientSAM-Tiny conversion flow for RDK X5. The runtime sample uses two quantized `.bin` models: image encoder and fixed-prompt decoder.

## Directory Structure

```text
.
|-- configs
|   |-- efficient_sam_vitt_encoder_featuremap_config.yaml
|   `-- efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
|-- scripts
|   |-- download_assets.py
|   |-- export_encoder_onnx.py
|   |-- export_decoder_onnx.py
|   |-- prepare_calibration.py
|   |-- prepare_efficient_decoder_calibration.py
|   `-- quantize.py
|-- README.md
|-- README_cn.md
|-- QUANTIZATION_STATUS.md
`-- VALIDATION.md
```

Generated ONNX files, `*_quant_info.json`, calibration tensors, and quantized `.bin` outputs are produced by the steps below and are not committed with the sample. Keep generated files in the conversion workspace or output folders, then use `model/download_model.sh` or copy locally generated `.bin` files into `../model/` when running the demo.

## 1. Clone Official Repository

Official source: https://github.com/yformer/EfficientSAM

```bash
cd samples/vision/efficient_sam/conversion
python3 scripts/download_assets.py --workspace ./workspace
```

This creates:

```text
workspace/EfficientSAM
workspace/EfficientSAM/weights/efficient_sam_vitt.pt
```

## 2. Export Encoder ONNX

The encoder export loads `weights/efficient_sam_vitt.pt`, fixes image size to `512x512`, and exports an opset-11 ONNX with split QKV attention for toolchain compatibility.

```bash
python3 scripts/export_encoder_onnx.py \
  --repo ./workspace/EfficientSAM \
  --weights ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_encoder_512_splitqkv_op11.onnx
```

Expected interface:

```text
input:  batched_images, 1x3x512x512, float32 NCHW
output: image_embeddings, 1x256x32x32, float32 NCHW
```

## 3. Export Fixed-Prompt Decoder ONNX

The decoder export loads the same checkpoint and bakes two positive point prompts into the ONNX so the board demo can run a static full-mask pipeline with one decoder input.

```bash
python3 scripts/export_decoder_onnx.py \
  --repo ./workspace/EfficientSAM \
  --weights ./workspace/EfficientSAM/weights/efficient_sam_vitt.pt \
  --output ./efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx
```

Expected interface:

```text
input:  image_embeddings, 1x256x32x32, float32 NCHW
outputs:
  low_res_masks, 1x3x128x128, float32
  iou_predictions, 1x3, float32
```

## 4. Prepare Calibration Data

Encoder calibration uses representative images and writes RGB CHW `1x3x512x512` float32 raw tensors:

```bash
python3 scripts/prepare_calibration.py \
  --src /path/to/calibration/images \
  --out ./calibration_data_rgbchw_512 \
  --num 30
```

Decoder calibration uses real encoder embeddings:

```bash
python3 scripts/prepare_efficient_decoder_calibration.py \
  --embedding /path/to/efficient_sam_image_embeddings_f32.bin \
  --out ./decoder_calibration \
  --num 30
```

## 5. Quantize with Matching YAML

Run inside OE Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` from this directory.

Encoder YAML:

```text
configs/efficient_sam_vitt_encoder_featuremap_config.yaml
```

It consumes `efficient_sam_vitt_encoder_512_splitqkv_op11.onnx` and `calibration_data_rgbchw_512`, then emits:

```text
bpu_model_output_512_default_none/efficient_sam_vitt_encoder_512x512_default_none.bin
```

Decoder YAML:

```text
configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
```

It consumes `efficient_sam_vitt_decoder_fixedprompt_512_op11.onnx` and decoder embedding calibration, then emits:

```text
bpu_model_output_decoder_fixedprompt/efficient_sam_vitt_decoder_fixedprompt_512_default.bin
```

Commands:

```bash
python3 scripts/quantize.py --config configs/efficient_sam_vitt_encoder_featuremap_config.yaml
python3 scripts/quantize.py --config configs/efficient_sam_vitt_decoder_fixedprompt_512_default_config.yaml
```

Copy the generated `.bin` files into `../model/` for local validation, or use `../model/download_model.sh` to fetch the published models.