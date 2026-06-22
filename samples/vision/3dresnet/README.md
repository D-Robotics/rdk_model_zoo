English | [简体中文](./README_cn.md)

# 3D ResNet-18 Model Description

3D ResNet-18 (R3D-18) is a video action classification model. It extends 2D ResNet with 3D convolution so that spatial and temporal features are learned from a short video clip. This sample runs a preprocessed 16-frame clip and prints Top-K Kinetics-400 action predictions.

The provided test clip is `video0.npy`. For this clip, a reasonable result should classify the action as `archery`.

## Algorithm Overview

R3D-18 extends ResNet18 from 2D convolution to 3D convolution, allowing the network to model spatial and temporal features in video clips.

- **Paper**: [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- **Reference Implementation**: [torchvision r3d_18](https://pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html)

### Algorithm Capabilities

- Kinetics-400 video action classification
- Top-K action label prediction

### Algorithm Features

- **3D convolution**: extracts features across spatial and temporal dimensions.
- **Residual learning**: follows the ResNet residual block design.
- **Prepared clip input**: uses the `.npy` video clip.

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
|       |-- resnet3d.py
|       `-- run.sh
`-- test_data
    |-- kinetics_classnames.json
    |-- readme_img
    `-- video0.npy
```

## Quick Start

```bash
cd runtime/python
bash run.sh
```

The script downloads `../../model/s100/r3d_18.hbm` if needed and runs inference on `../../test_data/video0.npy`.

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Model | Task | Input | Output |
| ----- | ---- | ----- | ------ |
| R3D-18 | Video action classification | `(1, 3, 16, 112, 112)` float32 clip | Kinetics-400 logits |

## Model Evaluation

Evaluation notes, performance records, and result checks are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

For `video0.npy`, the expected Top-1 class is `archery`, with significantly lower scores for the remaining Top-5 classes.

## License

Follow the top-level Model Zoo License.
