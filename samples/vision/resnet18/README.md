English | [简体中文](./README_cn.md)

# ResNet18 Model Description

ResNet18 is an ImageNet classification sample for the RDK S100 model zoo. It
provides sample-local model download, Python and C++ runtime examples, preserved
original documentation assets, and validation notes.

## Algorithm Overview

ResNet was proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
The key idea is residual learning with shortcut connections, which reduces the
optimization difficulty of deep convolutional networks and avoids degradation as
the network becomes deeper.

![ResNet architecture](./test_data/resnet_architecture.png)

Resources:

- Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- PyTorch implementation: [torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- TorchVision ResNet18 model: [torchvision ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)

### Algorithm Capabilities

- ImageNet 1000-class image classification
- Top-K class ID and confidence output

### Algorithm Features

- **Residual connections**: use shortcut connections to reduce deep-network optimization difficulty.
- **Lightweight residual network**: the 18-layer variant is suitable for quick classification validation.
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
|   |-- cpp
|   |   |-- CMakeLists.txt
|   |   |-- README.md
|   |   |-- README_cn.md
|   |   |-- inc
|   |   |   `-- resnet18.hpp
|   |   |-- run.sh
|   |   `-- src
|   |       |-- main.cpp
|   |       `-- resnet18.cpp
|   `-- python
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- resnet18.py
|       `-- run.sh
`-- test_data
    |-- resnet_architecture.png
    |-- resnet_architecture2.png
    |-- result.png
    `-- zebra_cls.jpg
```

## Quick Start

Python:

```bash
cd runtime/python
bash run.sh
```

C++:

```bash
cd runtime/cpp
bash run.sh
```

Direct Python entry:

```bash
python3 main.py \
  --model-path ../../model/s100/resnet18_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../../../../datasets/imagenet/imagenet_classes.names \
  --top-k 5
```

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains Python and C++ runtime paths:

- [runtime/python/README.md](./runtime/python/README.md)
- [runtime/cpp/README.md](./runtime/cpp/README.md)

| Model | Input | Runtime model |
| --- | --- | --- |
| ResNet18 | 224x224 NV12 | `model/s100/resnet18_224x224_nv12.hbm` |

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
