English | [简体中文](./README_cn.md)

# RepViT Model Description

This directory provides the complete usage guide for the RepViT sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

RepViT revisits lightweight mobile CNN design from a ViT perspective. It keeps a pure CNN deployment structure while borrowing lightweight ViT design ideas, and improves inference efficiency through structural reparameterization.

- **Paper**: [RepViT: Revisiting Mobile CNN From ViT Perspective](http://arxiv.org/abs/2307.09283)
- **Reference Implementation**: [THU-MIG/RepViT](https://github.com/THU-MIG/RepViT)

### Algorithm Functionality

RepViT supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **ViT-Inspired Mobile CNN**: Revisits MobileNet-style architecture from a lightweight ViT perspective.
- **Structural Reparameterization**: Fuses training-time structural branches for faster deployment-time inference.
- **Separated Mixers**: Separates token mixer and channel mixer to simplify the block structure.
- **Efficient Deployment**: Provides m0.9, m1.0, and m1.1 RDK X5 deployment models using packed NV12 input.

![RepViT Architecture](./test_data/RepViT_architecture.png)

![RepViT Depthwise Block](./test_data/RepViT_DW.png)

## Directory Structure

```text
.
|-- conversion
|   |-- RepViT_m0_9_config.yaml
|   |-- RepViT_m1_0_config.yaml
|   |-- RepViT_m1_1_config.yaml
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
|       |-- repvit.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- ImageNet_1k.json
|   |-- inference.png
|   |-- RepViT_architecture.png
|   |-- RepViT_DW.png
|   `-- yurt.JPEG
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

The following table shows the published RepViT performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RepViT-m1.1 | 224x224 | 1000 | 8.2 | 77.73% | 77.50% | 2.32 | 590.42 |
| RepViT-m1.0 | 224x224 | 1000 | 6.8 | 76.75% | 76.50% | 1.97 | 692.29 |
| RepViT-m0.9 | 224x224 | 1000 | 5.1 | 76.32% | 75.75% | 1.65 | 902.69 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
