# UNet-resnet50 Python Inference

[English](./README.md) | 简体中文

This directory contains the Python inference example for UNet-resnet50 semantic segmentation.

## Introduction

The UNet-resnet50 model performs pixel-level semantic segmentation, predicting a class label for each pixel in the input image. This example demonstrates how to load a BPU-quantized model, preprocess images, run inference, and visualize segmentation results.

## Environment Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Horizon `hobot_dnn` (pyeasy_dnn)

No additional Python packages are required beyond the standard Horizon runtime environment.

## Directory Structure

```bash
.
|-- unet.py      # Model implementation (Config + Model classes)
|-- main.py      # Main inference entry point
|-- run.sh       # One-click run script
`-- README.md    # This file
```

## Parameters

| Parameter      | Description                                          | Default Value                                              |
| -------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| `--model-path` | Path to the BPU quantized *.bin model file           | `./model/unet_resnet50_512x512_nv12.bin`                |
| `--img-path`   | Path to the input image for inference                | `./test_data/UNet_Segmentation_Origin.png`             |
| `--save-path`  | Path to save the visualization result                | `unet_result.jpg`                                          |
| `--mask-path`  | Path to save the raw segmentation mask               | `unet_mask.png`                                            |
| `--num-classes`| Number of segmentation classes (including background)| `21`                                                       |
| `--alpha`      | Opacity of the overlay mask in visualization         | `0.6`                                                      |

## Quick Start

### 1. Default Parameters (Zero Configuration)

If the model is placed at the system default path and test data exists:

```bash
bash run.sh
```

Or directly:

```bash
python3 main.py
```

### 2. Custom Parameters

```bash
python3 main.py \
    --model-path /path/to/unet_resnet50_512x512_nv12.bin \
    --img-path /path/to/image.jpg \
    --save-path result.jpg \
    --num-classes 21
```

### Output

- `unet_result.jpg` — Visualization result with color overlay and legend
- `unet_mask.png` — Raw segmentation mask (class indices scaled for visibility)

## Code Documentation

For detailed API documentation of the source code, please refer to the source reference documentation in the `docs/source_reference` directory.

## Notes

- The input image must be a valid 3-channel BGR image.
- The model expects NV12 input with resolution 512×512.
- The segmentation mask is resized back to the original image size using nearest-neighbor interpolation to preserve class boundaries.
