English | [简体中文](./README_cn.md)

# EfficientFormer Model Description

This directory provides the complete usage guide for the EfficientFormer sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

EfficientFormer is a vision transformer family designed for mobile-speed inference. It analyzes inefficient operators in ViT-style networks and introduces a latency-driven design that keeps transformer-style modeling while improving deployment efficiency on edge devices.

- **Paper**: [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)
- **Reference Implementation**: [snap-research/EfficientFormer](https://github.com/snap-research/EfficientFormer)

### Algorithm Functionality

EfficientFormer supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Latency-Driven Design**: Uses latency analysis to remove inefficient ViT operations for mobile inference.
- **Dimension-Consistent Blocks**: Keeps deployment-friendly tensor layouts for efficient execution.
- **Edge Deployment**: Provides L1 and L3 RDK X5 deployment models using packed NV12 input.

![Latency Profiling](./test_data/latency_profiling.png)

![EfficientFormer Architecture](./test_data/EfficientFormer_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- EfficientFormer_l1_config.yaml
|   |-- EfficientFormer_l3_config.yaml
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
|       |-- efficientformer.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- EfficientFormer_architecture.png
|   |-- ImageNet_1k.json
|   |-- bittern.JPEG
|   |-- inference.png
|   `-- latency_profiling.png
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

The following table shows the published EfficientFormer performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientFormer-L3 | 224x224 | 1000 | 31.3 | 76.75% | 76.05% | 17.55 | 60.52 |
| EfficientFormer-L1 | 224x224 | 1000 | 12.3 | 76.75% | 67.72% | 5.88 | 191.605 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
