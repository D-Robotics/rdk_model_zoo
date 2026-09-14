English | [简体中文](./README_cn.md)

# MobileSAM Conversion Guide

This directory documents the complete MobileSAM conversion flow for RDK X5. The runtime sample uses two quantized `.bin` models: image encoder and box-prompt decoder.

## Directory Structure

```text
.
|-- configs
|   |-- mobile_sam_image_encoder_norm_512x512_config.yaml
|   `-- mobile_sam_decoder_512_box_default_config.yaml
|-- scripts
|   |-- download_assets.py
|   |-- export_encoder_onnx.py
|   |-- export_decoder_onnx.py
|   |-- prepare_calibration.py
|   |-- prepare_decoder_calibration.py
|   `-- quantize.py
|-- README.md
`-- README_cn.md
```

Generated ONNX files, `*_quant_info.json`, calibration tensors, and quantized `.bin` outputs are produced by the steps below and are not committed with the sample. Keep generated files in the conversion workspace or output folders, then use `model/download_model.sh` or copy locally generated `.bin` files into `../model/` when running the demo.

## 1. Clone Official Repository

Official source: https://github.com/ChaoningZhang/MobileSAM

```bash
cd samples/vision/mobile_sam/conversion
python3 scripts/download_assets.py --workspace ./workspace
```

This creates:

```text
workspace/MobileSAM
workspace/MobileSAM/weights/mobile_sam.pt
```

## 2. Export Encoder ONNX

The encoder export loads `weights/mobile_sam.pt`, sets image size to `512x512`, applies MobileSAM normalization outside the model, and exports a fixed-shape opset-11 ONNX.

```bash
python3 scripts/export_encoder_onnx.py \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_image_encoder_norm_512_op11.onnx
```

Expected interface:

```text
input:  normalized_images, 1x3x512x512, float32 NCHW
output: image_embeddings, 1x256x32x32, float32 NCHW
```

## 3. Export Decoder ONNX

The decoder export loads the same checkpoint and exports a fixed box-prompt decoder with opset 11. The box prompt is supplied as a runtime input, so the board demo can change it through `--box`.

```bash
python3 scripts/export_decoder_onnx.py \
  --repo ./workspace/MobileSAM \
  --weights ./workspace/MobileSAM/weights/mobile_sam.pt \
  --output ./mobile_sam_decoder_512_box_op11.onnx
```

Expected interface:

```text
inputs:
  image_embeddings, 1x256x32x32, float32 NCHW
  boxes, 1x4, float32
outputs:
  low_res_masks, 1x3x128x128, float32
  iou_predictions, 1x3, float32
```

## 4. Prepare Calibration Data

Encoder calibration uses representative images and writes normalized `1x3x512x512` float32 raw tensors:

```bash
python3 scripts/prepare_calibration.py \
  --src /path/to/calibration/images \
  --out ./calibration_data_norm_512 \
  --num 30
```

Decoder calibration uses real encoder embeddings plus deterministic box jitter:

```bash
python3 scripts/prepare_decoder_calibration.py \
  --embedding /path/to/mobile_sam_image_embeddings_f32.bin \
  --out ./decoder_calibration \
  --num 30
```

## 5. Quantize with Matching YAML

Run inside OE Docker `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310` from this directory.

Encoder YAML:

```text
configs/mobile_sam_image_encoder_norm_512x512_config.yaml
```

It consumes `mobile_sam_image_encoder_norm_512_op11.onnx` and `calibration_data_norm_512`, then emits:

```text
bpu_model_output/mobile_sam_image_encoder_norm_512x512_allint16.bin
```

Decoder YAML:

```text
configs/mobile_sam_decoder_512_box_default_config.yaml
```

It consumes `mobile_sam_decoder_512_box_op11.onnx` and decoder calibration featuremaps, then emits:

```text
bpu_model_output_decoder_default/mobile_sam_decoder_512_box_default.bin
```

Commands:

```bash
python3 scripts/quantize.py --config configs/mobile_sam_image_encoder_norm_512x512_config.yaml
python3 scripts/quantize.py --config configs/mobile_sam_decoder_512_box_default_config.yaml
```

Copy the generated `.bin` files into `../model/`.

## 6. Output

Copy the generated encoder and decoder `.bin` files into `../model/` for runtime inference and performance validation.
