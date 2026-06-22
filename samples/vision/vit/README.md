English | [简体中文](./README_cn.md)

# Vision Transformer Model Description

Vision Transformer (ViT) applies the Transformer architecture to image classification by splitting an image into patches and modeling patch tokens with self-attention. This sample runs a CIFAR-10 ViT classifier with NV12 HBM models and prints Top-K classification results.

![ViT architecture](./test_data/readme_img/vitnet.png)

## Algorithm Overview

ViT splits an image into fixed-size patches and feeds the patch sequence into a Transformer Encoder for classification.

- **Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- **Reference Implementation**: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## Algorithm Capabilities

- CIFAR-10 image classification
- int8 and int16 model variants
- Top-K class ID and confidence output

## Algorithm Features

- **Patch tokens**: converts images into patch sequences.
- **Self-attention**: models global relationships with Transformer blocks.
- **NV12 input**: runtime feeds Y and UV planes as two HBM input tensors.

## Directory Structure

```text
vit/
|-- conversion/
|   |-- README.md
|   |-- README_cn.md
|   |-- config_vit_nv12.yaml
|   `-- hb_compile.log
|-- evaluator/
|   |-- README.md
|   `-- README_cn.md
|-- model/
|   |-- README.md
|   |-- README_cn.md
|   `-- download_model.sh
|-- runtime/
|   `-- python/
|       |-- README.md
|       |-- README_cn.md
|       |-- main.py
|       |-- run.sh
|       `-- vit.py
|-- test_data/
|-- README.md
`-- README_cn.md
```

## Quick Start

Download the default model into the sample-local model directory:

```bash
cd samples/vision/vit/model
bash download_model.sh s100 int8
```

Run the Python sample:

```bash
cd ../runtime/python
bash run.sh int8
```

Run the entry script directly:

```bash
python3 main.py \
  --model-path ../../model/s100/vit_cifar10_batch1_int8.hbm \
  --test-img ../../test_data/airplane_0000.png \
  --label-file ../../test_data/cifar10_classes.names \
  --top-k 5
```

## Model Conversion

- Prebuilt HBM models are provided through the [model](./model/README.md) directory.
- Conversion notes are available in [conversion/README.md](./conversion/README.md).

## Runtime

This sample currently maintains the Python runtime path. See [runtime/python/README.md](./runtime/python/README.md) for details.

| Model | Dataset | Input | Runtime input type | Download path |
| --- | --- | --- | --- | --- |
| ViT CIFAR-10 int8 | CIFAR-10 | 224x224 | NV12 Y/UV planes | `model/s100/vit_cifar10_batch1_int8.hbm` |
| ViT CIFAR-10 int16 | CIFAR-10 | 224x224 | NV12 Y/UV planes | `model/s100/vit_cifar10_batch1_int16.hbm` |

This sample uses public S100 HBM models downloaded into the sample-local `model/s100` directory.

## Model Evaluation

Evaluation notes and result-check methods are available in [evaluator/README.md](./evaluator/README.md).

## Inference Result

Using `airplane_0000.png`, a correct run should include `airplane` in the Top-5 results, with a reasonable non-zero confidence distribution.

## License

Follow the top-level Model Zoo License.
