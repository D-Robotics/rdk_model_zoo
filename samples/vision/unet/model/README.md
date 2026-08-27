[English](./README.md) | [简体中文](./README_cn.md)

# UNet Model Files

This directory provides prebuilt UNet ResNet `.bin` downloads for RDK X5
(`bayes-e`). Model binaries are not committed to Git; the download script saves
them in this directory.

## Available Models

| Backbone | File | Status | SHA256 |
| --- | --- | --- | --- |
| ResNet18 | `unet_resnet18_voc_512x512_nv12.bin` | Published | `d082ff055532081d14326d96fb2bb8ac85a0f1edc46e868cbbbea0259bc36b5f` |
| ResNet34 | `unet_resnet34_voc_512x512_nv12.bin` | Published | `9d758822b2de4d5aaa24b4c02479c9f742c4b4e4af075389d921c12396194ac0` |
| ResNet50 | `unet_resnet50_voc_512x512_nv12.bin` | Published | `22ea4eec82328d34dc963e091f3cb4e134c8a432d7c6f92d74522b998b7bd23a` |
| ResNet101 | `unet_resnet101_voc_512x512_nv12.bin` | Published | `04031417d3d4098bceac0bb1b731a7aed099dca5c8ee0cd30527c2f6494c7215` |
| ResNet152 | `unet_resnet152_voc_512x512_nv12.bin` | Published | `990855473e5411c2996bd7f161591dc7ba479402bcfe40c36d3fd2b10edbb32a` |

## Download

Download the published ResNet18 model by default:

```bash
cd samples/vision/unet/model
./download_model.sh
```

Direct download:
[unet_resnet18_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet18_voc_512x512_nv12.bin)

[unet_resnet34_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet34_voc_512x512_nv12.bin)

[unet_resnet50_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet50_voc_512x512_nv12.bin)

[unet_resnet101_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet101_voc_512x512_nv12.bin)

[unet_resnet152_voc_512x512_nv12.bin](https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_x5/unet/unet_resnet152_voc_512x512_nv12.bin)

One or more backbones can also be selected explicitly:

```bash
./download_model.sh resnet18
./download_model.sh resnet34 resnet101
./download_model.sh all
```

Every download is checked against its published SHA256 value before the
temporary file is moved into place. A failed download or checksum never replaces
an existing model.

These models are for RDK X5 only and are not compatible with RDK S100/S600. See
the [conversion guide](../conversion/README.md) to build a model locally.
