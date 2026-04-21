English | [简体中文](./README_cn.md)

# MobileNetV4 Model Description

This directory provides the complete usage guide for the MobileNetV4 sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

MobileNetV4 is a lightweight image classification model family that introduces Universal Inverted Bottleneck and Mobile Multi-Query Attention for efficient mobile and embedded deployment.

- **Paper**: [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
- **Reference Implementation**: [timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)

### Algorithm Functionality

MobileNetV4 supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Universal Inverted Bottleneck**: Unifies inverted bottleneck, ConvNeXt-style blocks, FFN-style blocks, and ExtraDW variants.
- **Mobile Multi-Query Attention**: Provides an attention structure optimized for mobile accelerators.
- **Model Variants**: This sample provides Conv-Small and Conv-Medium deployment models.

![MobileNetV4 Architecture](./test_data/MobileNetV4_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- MobileNetV4_medium.yaml
|   |-- MobileNetV4_small.yaml
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
|       |-- mobilenetv4.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- great_grey_owl.JPEG
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- MobileNetV4_architecture.png
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

The following table shows the published MobileNetV4 performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Medium | 224x224 | 1000 | 9.7 | 76.8% | 75.1% | 2.42 | 572+ |
| MobileNetV4-Conv-Small | 224x224 | 1000 | 3.8 | 70.8% | 68.8% | 1.18 | 1436+ |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
