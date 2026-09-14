English | [简体中文](./README_cn.md)

# EfficientNet Model Description

This directory provides the complete usage guide for the EfficientNet sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

EfficientNet is an image classification model family that balances CNN input resolution, depth, and width through compound scaling. The method improves accuracy and efficiency under fixed compute budgets by scaling the three dimensions together instead of tuning one dimension independently.

- **Paper**: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- **Reference Implementation**: [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

### Algorithm Functionality

EfficientNet supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Compound Scaling**: Jointly scales resolution, depth, and width to balance accuracy and efficiency.
- **AutoML Backbone Search**: Uses neural architecture search to obtain efficient baseline networks.
- **Efficient Deployment**: Provides B2, B3, and B4 RDK X5 deployment models using packed NV12 input.

![EfficientNet Architecture](./test_data/EfficientNet_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- EfficientNet_B2_config.yaml
|   |-- EfficientNet_B3_config.yaml
|   |-- EfficientNet_B4_config.yaml
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
|       |-- efficientnet.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- EfficientNet_architecture.png
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- redshank.JPEG
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

The following table shows the published EfficientNet performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientNet-B4 | 224x224 | 1000 | 19.27 | 74.25% | 71.75% | 5.44 | 212.75 |
| EfficientNet-B3 | 224x224 | 1000 | 12.19 | 76.22% | 74.05% | 3.96 | 310.30 |
| EfficientNet-B2 | 224x224 | 1000 | 9.07 | 76.50% | 73.25% | 3.31 | 376.77 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
