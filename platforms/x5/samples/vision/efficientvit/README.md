English | [简体中文](./README_cn.md)

# EfficientViT Model Description

This directory provides the complete usage guide for the EfficientViT sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

EfficientViT_MSRA is a lightweight vision transformer family designed around memory-efficient attention and deployment-friendly building blocks. It reduces the runtime bottlenecks of standard self-attention while preserving competitive ImageNet classification accuracy.

- **Paper**: [EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/abs/2305.07027)
- **Reference Implementation**: [microsoft/Cream - EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)

### Algorithm Functionality

EfficientViT supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Memory-efficient attention**: Reduces the data-movement overhead that limits standard transformer inference efficiency.
- **Cascaded group attention**: Improves representation capability while controlling deployment cost.
- **Deployment-friendly normalization**: Uses batch normalization to simplify inference-side fusion.
- **Classification output**: Produces Top-K class IDs and confidence scores for ImageNet-1k labels.

![Transformer vs CNN](./test_data/comparison_between_transformer_and_cnn.png)
![MHSA Computation](./test_data/mhsa_computation.jpg)
![EfficientViT Architecture](./test_data/efficientvit_msra_architecture.png)

## Directory Structure

```text
.
|-- conversion
|   |-- EfficientViT_MSRA_m5_config.yaml
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
|       |-- efficientvit.py
|       |-- main.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- comparison_between_transformer_and_cnn.png
|   |-- efficientvit_msra_architecture.png
|   |-- hook.JPEG
|   |-- ImageNet_1k.json
|   |-- inference.png
|   `-- mhsa_computation.jpg
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

The following table shows the published EfficientViT performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
