English | [简体中文](./README_cn.md)

# MobileNetV4 Model Description

MobileNetV4 is an ImageNet classification sample for the RDK model zoo,
supporting both S100 and S600 platforms via SOC-aware HBM download.
It provides a standard Python runtime based on `hbm_runtime`, sample-local model
download, preserved conversion assets, and validation notes.

## Algorithm Overview

MobileNetV4 introduces the Universal Inverted Bottleneck (UIB) search block and
mobile-friendly attention designs for efficient image classification on mobile
and embedded accelerators.

Resources:

- Paper: [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518)
- timm implementation: [huggingface/pytorch-image-models MobileNetV4](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py)

### Algorithm Capabilities

- ImageNet 1000-class image classification
- Small and Medium model variants
- Top-K class ID and confidence output

### Algorithm Features

- **UIB module**: unifies inverted bottleneck, conv-next style, and FFN-style structures.
- **Mobile attention design**: balances accuracy and efficiency for mobile and embedded inference.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

## Directory Structure

```text
.
|-- README.md
|-- README_cn.md
|-- conversion
|   |-- README.md
|   |-- README_cn.md
|   |-- get_calibration_data.py
|   |-- get_mobilenetv4_onnx.py
|   |-- mobilenetv4_medium_config.yaml
|   |-- mobilenetv4_small_config.yaml
|   |-- timm2onnx_local.py
|   `-- x86_medium_inference.py
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
|       |-- mobilenetv4.py
|       `-- run.sh
`-- test_data
    |-- imagenet_classes.names
    `-- zebra_cls.jpg
```

## Quick Start

```bash
cd runtime/python
bash run.sh
```

Run the medium model:

```bash
bash run.sh medium
```

Direct entry (substitute `<soc>` with `s100` or `s600`):

```bash
python3 main.py \
  --model-variant small \
  --model-path ../../model/<soc>/mobilenetv4_small_224x224_nv12.hbm \
  --test-img ../../test_data/zebra_cls.jpg \
  --label-file ../../test_data/imagenet_classes.names \
  --top-k 5
```

Expected result for the included test image:

```text
Top-5 Classification Results:
  [0] zebra: ...
```

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Variant | Input | Runtime model |
| --- | --- | --- |
| Small | 224x224 NV12 | `model/<soc>/mobilenetv4_small_224x224_nv12.hbm` |
| Medium | 256x256 NV12 | `model/<soc>/mobilenetv4_medium_256x256_nv12.hbm` |

`<soc>` is `s100` or `s600`, resolved automatically from `/sys/class/boardinfo/soc_name`.

## Model Evaluation

Evaluation notes and result-check methods are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

Using `zebra_cls.jpg`, a correct run should include `zebra` in the Top-5 results, with a reasonable non-zero confidence distribution.

## License

Follow the top-level Model Zoo License.
