English | [简体中文](README_cn.md)

# HGNetV2 image classification

HGNetV2 is a convolutional backbone for image classification.

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

HGNetV2 is a convolutional backbone for vision tasks; this sample exposes
b0–b4 ImageNet-1k classification models. HGNetV2 is a next-generation CNN
backbone designed for a strong accuracy/latency balance, succeeding the
original HGNet, and performing well in classification, detection and
segmentation.

[PP-HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/en/models/ImageNet1k/PP-HGNetV2.md)

Feature highlights:

- **Aggregating multiple receptive fields** — the HG-Block combines
  multi-scale features from shallow to deep layers, which is friendly to
  small-object detection and recognition.
- **Improved stem module** — the input stem stacks more 2×2 convolution
  kernels to learn rich local features while using smaller channel
  numbers, boosting performance on high-resolution tasks.
- **Learnable downsampling (LDS)** — an adaptive downsampling layer
  preserves more useful spatial details while reducing computational
  redundancy.

One BGR image produces ImageNet-1k Top-K class IDs, scores and optional
labels. The `HGNetV2Classifier` class runs a `preprocess → infer →
postprocess` flow chained by `predict` (labels, drawing and file output
belong to the CLI layer; see
[runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
hgnetv2/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python inference
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Python dependencies
```

<a id="support-matrix"></a>
## Support matrix

The table lists the models published for this sample; `—` marks an
unpublished combination.

| Model | Input | X5 | S100 | S100P | S600 |
| --- | --- | --- | --- | --- | --- |
| HGNetV2-B0 | 224x224 | ✓ | — | — | — |
| HGNetV2-B1 | 224x224 | ✓ | — | — | — |
| HGNetV2-B2 | 224x224 | ✓ | — | — | — |
| HGNetV2-B3 | 224x224 | ✓ | — | — | — |
| HGNetV2-B4 | 224x224 | ✓ | — | — | — |

Runtime coverage: the Python CLI serves every published combination with
`--target auto|x5` and `--variant b0|b1|b2|b3|b4` (default `b0`); the
lowercase CLI IDs map to the exact published filenames
`hgnetv2_b<0-4>_224x224_nv12.bin` (letter casing in filenames is
preserved). `--target s100`, `s100p` and `s600` resolve no asset. The
sample ships no C++ runtime.

<a id="prerequisites"></a>
## Prerequisites

Use a full repository checkout. On X5 use the matching board image and
its `hbm_runtime`; install the host dependencies in a virtual environment
as below. Host-side comparison tools use SciPy; board inference uses the dependencies in the matching runtime image. Native inference needs no OE toolchain; conversion
prerequisites are documented under [conversion](conversion/README.md).

```bash
# cwd: repository root
python3 -m venv .venv-hgnetv2
source .venv-hgnetv2/bin/activate
python3 -m pip install -r samples/vision/hgnetv2/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

From the repository root, prepare the X5 artifact with the model downloader, then run inference with the bundled image and the selected artifact reference.

```bash
# cwd: repository root
bash samples/vision/hgnetv2/model/download.sh x5 b0
python3 samples/vision/hgnetv2/runtime/python/main.py \
  --target x5 --variant b0 \
  --test-img samples/vision/hgnetv2/test_data/sandbar.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

<a id="expected-results"></a>
## Expected results

The default variant is `b0`; choose `b1`, `b2`, `b3`, `b4` explicitly.
Softmax scores produce a stable Top-K, with exact ties ordered by
ascending class ID. `sandbar.JPEG` is the bundled functional input; use the evaluator guide for dataset accuracy. Pass `--img-save-path` to save an image; otherwise results are printed to stdout.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `sandbar.JPEG`, with rank 1
being class 977 (sandbar, sand bar).

![Reference inference result on X5: sandbar test image with the top-5
overlay, rank 1 class 977 (sandbar, sand bar)](./test_data/result.jpg)

<a id="performance"></a>
## Performance data

X5 records from the x5-v1.1.3 release; full conditions and all rows are in
[evaluator](evaluator/README.md#reference-results).

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Single-thread latency (ms) | Multi-thread latency (ms) | FPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HGNetv2_b0 | 224x224 | 6.0 | 77.342 | 72.17 | 1.96 | 3.29 | 902.09 |
| HGNetv2_b1 | 224x224 | 6.34 | 78.872 | 73.47 | 2.41 | 3.89 | 760.13 |
| HGNetv2_b2 | 224x224 | 11.2 | 81.578 | 75.55 | 3.52 | 7.41 | 401.16 |
| HGNetv2_b3 | 224x224 | 16.3 | 82.916 | 76.51 | 4.53 | 10.37 | 287.27 |
| HGNetv2_b4 | 224x224 | 19.8 | 83.694 | 81.93 | 5.29 | 12.32 | 241.94 |

<a id="entry-points"></a>
## Entry points

[Model](model/README.md) · [Python](runtime/python/README.md) ·
[Conversion](conversion/README.md) · [Evaluation](evaluator/README.md)

<a id="license"></a>
## License

Python code follows Apache-2.0. Follow the original conversion notices and applicable upstream model and weight license terms. Review the upstream model and weight license terms before redistribution.

<!-- converted-batch-20261102 -->
## Converted-only additions (2026-11-02)

These independently converted timm weights have passed strict checkpoint loading,
reparameterization and two-input PyTorch/ONNX host comparison, checker and compiler.
Board smoke, accuracy, Runtime 1/2 and C++ 1/2 are **not-run**; historical metrics
and screenshots do not describe these artifacts. Existing defaults are retained.
Pillow is needed for the explicit bicubic shorter-edge 256 / rounded center crop
224 profile. Normalization is fused in Mapper from the actual checkpoint config;
output is float32 1,000-class logits followed by native softmax.

| Variant | Artifact | Checkpoint revision | Board |
|---|---|---|---|
| `timm-b0-ssld-stage2-ft-in1k` | `hgnetv2_b0_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`9e0a904b210d`](https://huggingface.co/timm/hgnetv2_b0.ssld_stage2_ft_in1k/blob/9e0a904b210d65e4fcd6582f8ca6e0921fac2650/README.md) | not-run |
| `timm-b1-ssld-stage2-ft-in1k` | `hgnetv2_b1_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`ec7014d7f1d0`](https://huggingface.co/timm/hgnetv2_b1.ssld_stage2_ft_in1k/blob/ec7014d7f1d0dd1ea2bf3012bc51dc95c5919fd2/README.md) | not-run |
| `timm-b2-ssld-stage2-ft-in1k` | `hgnetv2_b2_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`7a3092bc8711`](https://huggingface.co/timm/hgnetv2_b2.ssld_stage2_ft_in1k/blob/7a3092bc87118c7974bb9000b1eb6d9c560f8f75/README.md) | not-run |
| `timm-b3-ssld-stage2-ft-in1k` | `hgnetv2_b3_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`1aac877b3ea5`](https://huggingface.co/timm/hgnetv2_b3.ssld_stage2_ft_in1k/blob/1aac877b3ea55ab0f06d57cad9889d1e601bd472/README.md) | not-run |
| `timm-b4-ssld-stage2-ft-in1k` | `hgnetv2_b4_ssld_stage2_ft_in1k_bayese_224x224_nv12.bin` | [`3c808d9d9d98`](https://huggingface.co/timm/hgnetv2_b4.ssld_stage2_ft_in1k/blob/3c808d9d9d9873ed5e37b00509ba2a4882e816e8/README.md) | not-run |

```bash
python3 samples/vision/hgnetv2/model/download.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k
python3 samples/vision/hgnetv2/runtime/python/main.py --target x5 --variant timm-b0-ssld-stage2-ft-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

Independent converted profiles and explicit commands are listed in [model preparation](model/README.md). Board accuracy and performance are not measured yet.
