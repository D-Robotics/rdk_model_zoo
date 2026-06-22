English | [简体中文](./README_cn.md)

# MobileNetV3 Model Description

MobileNetV3 is a lightweight ImageNet classification model that combines neural
architecture search, NetAdapt, squeeze-and-excitation modules, and hard-swish
activation. This sample provides a Python runtime for RDK S-series
devices with `hbm_runtime` and NV12 model input.

## Algorithm Overview

MobileNetV3 is an embedded-oriented lightweight CNN that combines NAS search, NetAdapt, squeeze-and-excitation modules, and hard-swish activation.

- **Paper**: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **Reference Implementation**: [torchvision MobileNetV3](https://pytorch.org/vision/main/models/mobilenetv3.html)

### Algorithm Capabilities

- ImageNet 1000-class image classification
- Top-K class ID and confidence output

### Algorithm Features

- **NAS searched architecture**: combines automated search and manual optimization.
- **SE and hard-swish**: improves representation and mobile inference efficiency.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

## Directory Structure

```text
.
|-- conversion/             # Original conversion YAML and helper scripts
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
  --model-path ../../model/s100/mobilenetv3_224x224_nv12.hbm \
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
| MobileNetV3-Large | Image classification | 224x224 NV12 (Y + UV) | ImageNet 1000 | S100 |

This sample uses the public S100 HBM model downloaded into the sample-local
`model/s100/` directory.

## Model Evaluation

Evaluation notes and result-check methods are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

Using `zebra_cls.jpg`, a correct run should include `zebra` in the Top-5 results,
with a reasonable non-zero confidence distribution.

## License

Follow the top-level Model Zoo License.
