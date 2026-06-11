English | [简体中文](./README_cn.md)

# ResNet50 Model Description

ResNet50 is an ImageNet classification sample for the RDK S100 model zoo. It
provides sample-local model download, a Python runtime example, preserved
original documentation assets, and evaluation notes.

## Algorithm Overview

ResNet uses residual learning with shortcut connections to reduce the
optimization difficulty of deep convolutional networks. ResNet50 uses bottleneck
residual blocks with `1x1`, `3x3`, and `1x1` convolutions to build a deeper
network with controlled computation.

![ResNet architecture](./test_data/resnet_architecture.png)

Resources:

- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch implementation: [torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- TorchVision ResNet50 model: [torchvision ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

### Algorithm Capabilities

- ImageNet 1000-class image classification
- Top-K class ID and confidence output

### Algorithm Features

- **Residual connections**: use shortcut connections to reduce deep-network optimization difficulty.
- **Bottleneck blocks**: improve feature representation efficiency in deeper networks.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   `-- README_cn.md
|-- evaluator
|   |-- README.md
|   `-- README_cn.md
|-- model
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime
|   `-- python
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- resnet50.py
|       `-- run.sh
`-- test_data
    |-- resnet_architecture.png
    |-- resnet_architecture2.png
    |-- result.png
    `-- zebra_cls.jpg
```

## Quick Start

```bash
cd runtime/python
bash run.sh
```

Direct Python entry:

```bash
python3 main.py \
  --model-path ../../model/s100/resnet50_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Model | Input | Runtime model |
| --- | --- | --- |
| ResNet50 | 224x224 NV12 | `model/s100/resnet50_224x224_nv12.hbm` |

## Model Evaluation

Evaluation notes and result-check methods are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

The included test image:

![Test Image](./test_data/zebra_cls.jpg)

Expected Top-5 classification results:

```text
Top-5 Classification Results:
  [0] zebra: ...
```

Visualization of the classification result:

![Inference Result](./test_data/result.png)

## License

Follow the top-level Model Zoo License.
