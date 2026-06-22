English | [简体中文](./README_cn.md)

# MobileNetV1 Model Description

MobileNetV1 is a lightweight ImageNet classification model based on depthwise
separable convolutions. This sample provides a Python runtime for
RDK S-series devices with `hbm_runtime` and NV12 model input.

## Algorithm Overview

MobileNetV1 is a lightweight convolutional neural network for efficient image classification on embedded and mobile devices.

- **Paper**: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- **Reference Implementation**: [tensorflow/models MobileNetV1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)

### Algorithm Capabilities

- ImageNet 1000-class image classification
- Top-K class ID and confidence output

### Algorithm Features

- **Depthwise separable convolution**: decomposes standard convolution into depthwise and pointwise convolution.
- **Lightweight design**: reduces parameters and computation for edge deployment.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

## Directory Structure

```text
.
|-- conversion/             # Model conversion notes
|-- evaluator/              # Accuracy and result validation notes
|-- model/                  # HBM download script and model README
|-- runtime/
|   `-- python/             # Python runtime entry and wrapper
|-- test_data/              # Test image and ImageNet labels
|-- README.md
`-- README_cn.md
```

## Quick Start

```bash
cd runtime/python
bash run.sh
```

The script downloads the published S100 HBM model to `model/s100/` and runs
classification on `test_data/zebra_cls.jpg`.

For direct execution:

```bash
cd runtime/python
python3 main.py \
  --model-path ../../model/s100/mobilenetv1_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names
```

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Model | Task | Input | Classes | Published HBM |
| --- | --- | --- | --- | --- |
| MobileNetV1 | Image classification | 224x224 NV12 (Y + UV) | ImageNet 1000 | S100 |

This sample uses the public S100 HBM model downloaded into the sample-local
`model/s100/` directory.

## Model Evaluation

Evaluation notes and result-check methods are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

Using `zebra_cls.jpg`, a correct run should include `zebra` in the Top-5 results,
with a reasonable non-zero confidence distribution.

## License

Follow the top-level Model Zoo License.
