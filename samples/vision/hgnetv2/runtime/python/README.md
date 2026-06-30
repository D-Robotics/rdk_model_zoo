# HGNetV2 Image Classification Python Example

English | [简体中文](./README_cn.md)

This example demonstrates how to perform ImageNet-1k image classification tasks on the BPU using a quantized HGNetV2 model.

## Directory Structure

```text
.
|-- main.py
|-- hgnetv2.py
|-- README.md
|-- README_cn.md
`-- run.sh
```

## Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--model-path` | Path to the quantized `.bin` model file. | `../../model/hgnetv2_b0_224x224_nv12.bin` |
| `--label-file` | Path to the ImageNet label file. | `../../../../../datasets/imagenet/imagenet_classes.names` |
| `--priority` | Model priority, range `0~255`. | `0` |
| `--bpu-cores` | BPU core index used for inference. | `0` |
| `--test-img` | Path to the test input image. | `../../test_data/sandbar.JPEG` |
| `--img-save-path` | Path to save the output visualization image. | `../../test_data/result.jpg` |
| `--resize-type` | Resize strategy (`0`: direct resize, `1`: letterbox). | `0` |
| `--topk` | Number of top-K categories to display. | `5` |

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

## Manual Execution

- Using default parameters:

```bash
python3 main.py
```

- Explicitly specifying parameters:

```bash
python3 main.py \
    --model-path ../../model/hgnetv2_b0_224x224_nv12.bin \
    --test-img ../../test_data/sandbar.JPEG \
    --img-save-path ../../test_data/result.jpg \
    --topk 5
```

- Switch to a different variant (b1..b4):

```bash
python3 main.py --model-path ../../model/hgnetv2_b4_224x224_nv12.bin
```

## API Description

- **HGNetV2Config**: Encapsulates the model path, label file, and inference parameters.
- **HGNetV2**: Implements preprocessing, BPU inference, and top‑K classification post‑processing.