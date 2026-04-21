English | [简体中文](./README_cn.md)

# MobileNetV2 Model Description

This directory provides the complete usage guide for the MobileNetV2 sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

MobileNetV2 is a lightweight convolutional neural network that introduces inverted residual blocks and linear bottlenecks for efficient image classification.

- **Paper**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- **Reference Implementation**: [timm/models/mobilenetv2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv2.py)

### Algorithm Functionality

MobileNetV2 supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Inverted Residuals**: Expands channels before depthwise convolution and projects them back through a linear bottleneck.
- **Depthwise Separable Convolution**: Reduces computation compared with standard convolution.
- **Classification Output**: Outputs Top-K class IDs and confidence scores for ImageNet-1k labels.

![MobileNetV2 Architecture](./test_data/mobilenetv2_architecture.png)

## Directory Structure

```text
.
|-- conversion
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
|       |-- mobilenetv2.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- ImageNet_1k.json
|   |-- inference.png
|   |-- mobilenetv2_architecture.png
|   |-- Scottish_deerhound.JPEG
|   `-- seperated_conv.png
|-- README.md
`-- README_cn.md
```

## QuickStart

### Python

- Go to [runtime/python/README.md](./runtime/python/README.md) for detailed Python usage.
- For a quick experience:

```bash
cd runtime/python
bash run.sh
```

## Model Conversion

- Prebuilt `.bin` model files are provided through the [model](./model/README.md) directory.
- Conversion guidance is provided in [conversion/README.md](./conversion/README.md).

## Runtime Inference

The maintained inference path for this sample is Python.

- Python runtime guide: [runtime/python/README.md](./runtime/python/README.md)

## Evaluator

Evaluation notes, performance data, and validation summary are provided in [evaluator/README.md](./evaluator/README.md).

## Performance Data

The following table shows the published MobileNetV2 performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV2 | 224x224 | 1000 | 3.4 | 72.0% | 68.17% | 1.42 | 1152.07 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
