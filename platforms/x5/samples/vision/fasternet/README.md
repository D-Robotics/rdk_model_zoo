English | [简体中文](./README_cn.md)

# FasterNet Model Description

This directory provides the complete usage guide for the FasterNet sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

FasterNet is a lightweight CNN family designed around the idea of achieving higher effective FLOPS instead of only reducing theoretical FLOPs. It uses partial convolution to reduce redundant memory access and improve practical runtime efficiency on edge devices.

- **Paper**: [Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks](http://arxiv.org/abs/2303.03667)
- **Reference Implementation**: [JierunChen/FasterNet](https://github.com/JierunChen/FasterNet)

### Algorithm Functionality

FasterNet supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **High-FLOPS Design**: Emphasizes practical compute efficiency instead of minimizing theoretical FLOPs only.
- **Partial Convolution**: Uses PConv to reduce redundant computation and memory access.
- **Lightweight CNN Backbone**: Maintains a deployment-friendly CNN structure for efficient board-side inference.
- **Efficient Deployment**: Provides S, T0, T1, and T2 RDK X5 deployment models using packed NV12 input.

![FLOPs of Networks](./test_data/FLOPs%20of%20Nets.png)

![FasterNet Architecture](./test_data/FasterNet_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- FasterNet_S_config.yaml
|   |-- FasterNet_T0_config.yaml
|   |-- FasterNet_T1_config.yaml
|   |-- FasterNet_T2_config.yaml
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
|       |-- fasternet.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- drake.JPEG
|   |-- FasterNet_architecture.png
|   |-- FLOPs of Nets.png
|   |-- ImageNet_1k.json
|   `-- inference.png
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

The following table shows the published FasterNet performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 1000 | 31.1 | 77.04% | 76.15% | 6.73 | 162.83 |
| FasterNet-T2 | 224x224 | 1000 | 15.0 | 76.50% | 76.05% | 3.39 | 342.48 |
| FasterNet-T1 | 224x224 | 1000 | 7.6 | 74.29% | 71.25% | 1.96 | 708.40 |
| FasterNet-T0 | 224x224 | 1000 | 3.9 | 71.75% | 68.50% | 1.41 | 1135.13 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
