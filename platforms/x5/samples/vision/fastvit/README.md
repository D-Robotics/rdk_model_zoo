English | [简体中文](./README_cn.md)

# FastViT Model Description

This directory provides the complete usage guide for the FastViT sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

FastViT is a hybrid vision transformer family that uses structural reparameterization to build efficient token mixing blocks. The model combines convolutional stages and attention blocks to improve inference efficiency while keeping ImageNet classification accuracy competitive.

- **Paper**: [FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](http://arxiv.org/abs/2303.14189)
- **Reference Implementation**: [apple/ml-fastvit](https://github.com/apple/ml-fastvit)

### Algorithm Functionality

FastViT supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **RepMixer Token Mixing**: Uses structural reparameterization to reduce memory access overhead.
- **Hybrid Architecture**: Combines convolutional operations and attention to balance accuracy and efficiency.
- **Efficient Deployment**: Provides T8, T12, S12, and SA12 RDK X5 deployment models using packed NV12 input.

![FastViT Architecture](./test_data/FastViT_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- FastViT_S12_config.yaml
|   |-- FastViT_SA12_config.yaml
|   |-- FastViT_T12_config.yaml
|   |-- FastViT_T8_config.yaml
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
|       |-- fastvit.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- FastViT_architecture.png
|   |-- ImageNet_1k.json
|   |-- bucket.JPEG
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

The following table shows the published FastViT performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FastViT-SA12 | 224x224 | 1000 | 10.9 | 78.25% | 74.50% | 11.56 | 93.44 |
| FastViT-S12 | 224x224 | 1000 | 8.8 | 76.50% | 72.00% | 5.86 | 193.87 |
| FastViT-T12 | 224x224 | 1000 | 6.8 | 74.75% | 70.43% | 4.97 | 234.78 |
| FastViT-T8 | 224x224 | 1000 | 3.6 | 73.50% | 68.50% | 2.09 | 667.21 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
