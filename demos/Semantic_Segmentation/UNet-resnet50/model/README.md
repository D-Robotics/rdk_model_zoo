# UNet-resnet50 Model Download

English | [简体中文](./README_cn.md)

## Pre-converted Model

Pre-converted BPU models for UNet-resnet50 are available for download. These models have been optimized for Horizon platforms and can be used directly for inference.

## Model Naming

Model files follow the naming convention:

```
<model_name>_<input_resolution>_<chip_name>.bin
```

Example:

```
unet_resnet50_512x512_x3.bin
unet_resnet50_512x512_x5.bin
```

## Download

Please obtain the model from the following sources:

1. **Model Zoo official download page** (recommended)
2. **System default path** on the target device:
   - RDK X3 / X3 Module: `/opt/hobot/model/basic/unet_resnet50_512x512_x3.bin`
   - RDK X5 / X5 Module: `/opt/hobot/model/basic/unet_resnet50_512x512_x5.bin`

## Model Information

| Attribute        | Value                         |
| ---------------- | ----------------------------- |
| Model            | UNet-resnet50                 |
| Input Resolution | 512×512                       |
| Input Format     | NV12                          |
| Output Shape     | [1, 21, 512, 512]             |
| Output Type      | INT32 (with dequantize scale) |
| Classes          | 21 (VOC format, background=0) |
