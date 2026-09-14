English | [简体中文](./README_cn.md)

# VargConvNet Model Description

This directory provides the complete usage guide for the VargConvNet sample in Model Zoo, including algorithm overview, model conversion, runtime inference, model file management, and evaluation notes.

## Algorithm Overview

VargConvNet is a lightweight convolutional classification model used for ImageNet-1k image classification on edge devices. The RDK X5 sample provides a prebuilt packed-NV12 `.bin` model and a Python runtime based on `hbm_runtime`.

### Algorithm Functionality

VargConvNet supports the following task:

- ImageNet 1000-class image classification

### Algorithm Features

- Lightweight CNN classification model.
- Single packed-NV12 input for RDK X5 deployment.
- Standard Top-K ImageNet classification output.

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
|       |-- vargconvnet.py
|       |-- README.md
|       |-- README_cn.md
|       `-- run.sh
|-- test_data
|   |-- box_turtle.JPEG
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

Evaluation notes and validation summary are provided in [evaluator/README.md](./evaluator/README.md).

## Performance Data

The original VargConvNet demo does not provide a published benchmark table. Use [evaluator/README.md](./evaluator/README.md) for validation notes.

![Inference Result](./test_data/inference.png)

## License

Follows the Model Zoo top-level License.
