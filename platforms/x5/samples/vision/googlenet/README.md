English | [简体中文](./README_cn.md)

# GoogLeNet Model Description

This directory provides the complete usage guide for the GoogLeNet sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

GoogLeNet is an image classification network based on the Inception module. It won the ImageNet classification challenge in 2014 and introduced a practical multi-branch structure for extracting features at different receptive fields.

- **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- **Reference Implementation**: [torchvision/models/googlenet.py](https://github.com/pytorch/vision/blob/main/torchvision/models/googlenet.py)

### Algorithm Functionality

GoogLeNet supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- **Inception Module**: Extracts multi-scale features using parallel convolution and pooling branches.
- **Parameter Efficiency**: Reduces model parameters compared with wider dense CNN designs.
- **Deep Architecture**: Uses a 22-layer classification backbone with efficient branch aggregation.
- **Embedded Deployment**: The RDK X5 deployment model uses packed NV12 input and a quantized `.bin` artifact.

![GoogLeNet Architecture](./test_data/GoogLeNet_architecture.png)

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
|       |-- googlenet.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- GoogLeNet_architecture.png
|   |-- ImageNet_1k.json
|   |-- indigo_bunting.JPEG
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

The following table shows the published GoogLeNet performance on `RDK X5`.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GoogLeNet | 224x224 | 1000 | 6.81 | 68.72% | 67.71% | 2.19 | 626.27 |

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
