# HGNetV2 Model Description

English | [简体中文](./README_cn.md)

This directory provides complete usage instructions for the HGNetV2 sample in the Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation instructions.

## Algorithm Introduction

HGNetV2 is a next‑generation convolutional neural network (CNN) backbone designed to achieve the best balance between accuracy and latency on NVIDIA GPUs. Building upon the original HGNet, HGNetV2 achieves fast inference speed while maintaining high accuracy, and performs excellently in tasks such as image classification, object detection, and segmentation, making it an ideal choice for GPU‑based computer vision applications.

- **Detailed Introduction**: [docs/en/models/ImageNet1k/PP-HGNetV2.md](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models/ImageNet1k/PP-HGNetV2.md)

### Algorithm Functions

HGNetV2 supports the following tasks:

- ImageNet 1000‑class image classification

### Algorithm Features

- **Aggregating multiple receptive fields**: The HG‑Block combines multi‑scale features, capturing feature information of different sizes from shallow to deep layers, which is friendly to small object detection and recognition.
- **Improved stem module**: The initial preprocessing layers of the network are improved by stacking more \(2 \times 2\) convolution kernels to learn rich local features, while using smaller channel numbers, boosting performance on high‑resolution tasks.
- **Learnable downsampling (LDS)**: Integrates an adaptive downsampling layer that preserves more useful spatial details while reducing computational redundancy.

## Directory Structure

```text
.
|-- conversion
|   |-- HGNetV2_medium.yaml
|   |-- HGNetV2_small.yaml
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- download.sh
|   |-- README.md
|   `-- README_cn.md
|-- runtime
|   `-- python
|       |-- main.py
|       |-- HGNetV2.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- sandbar.JPEG
|   |-- classname.txt
|   `-- result.png
|-- README.md
`-- README_cn.md
```

## Quick Start

### Python

- For detailed Python instructions, please refer to [runtime/python/README.md](./runtime/python/README.md).
- Quick start command:

```bash
cd runtime/python
bash run.sh
```

## Model Conversion

- Pre‑compiled `.bin` models are provided via the [model](./model/README.md) directory.
- Conversion instructions can be found in [conversion/README.md](./conversion/README.md).

## Model Inference

Currently, this sample maintains the Python inference path.

- Python inference instructions: [runtime/python/README.md](./runtime/python/README.md)

## Model Evaluation

For evaluation instructions, performance data, and validation results, please refer to [evaluator/README.md](./evaluator/README.md).

## Performance Data

The following table shows the HGNetV2 performance data released on the `RDK X5`.

| Model | Input Size | Params (M) | Float Top-1 | Quantized Top-1 | Single‑thread Latency (ms) | Multi‑thread Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

![Inference result](./test_data/result.jpg)

## License

Follows the top‑level License of the Model Zoo.