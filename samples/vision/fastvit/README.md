English | [简体中文](README_cn.md)

# FastViT image classification

FastViT combines convolution and attention with structural reparameterization for image classification.

Sources: [FastViT: A Fast Hybrid Vision
Transformer using Structural
Reparameterization](https://arxiv.org/abs/2303.14189)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

FastViT is a hybrid vision transformer family that uses structural
reparameterization to build efficient token-mixing blocks: training-time
blocks keep their skip connections and extra branches, which are then
folded into plain convolutions for inference, cutting memory access
overhead. The token mixers differ across the family: in the upstream
model definitions, the published t8/t12/s12 variants use RepMixer token
mixing in all four stages, while sa12 keeps RepMixer in stages 1–3 and
uses self-attention (with a RepCPE positional encoding) only in stage 4
— a conv/attention hybrid configuration. ImageNet-1k
1000-class classification accuracy stays competitive. Feature
highlights:

- **RepMixer token mixing** — uses structural reparameterization to
  reduce memory access overhead.
- **Hybrid architecture** — combines convolutional operations and
  attention to balance accuracy and efficiency.
- **Efficient deployment** — T8, T12, S12, and SA12 RDK X5 deployment
  models with packed NV12 input.

![FastViT architecture: train-time vs inference-time structures, stem,
ConvFFN and RepMixer](./test_data/FastViT_architecture.png)

*Figure (upstream paper Fig. 2): (a) FastViT overview with decoupled
train-time and inference-time architectures; (b) convolutional stem;
(c) convolutional FFN; (d) RepMixer, which reparameterizes a skip
connection at inference. The paper draws stage 4 with a self-attention
token mixer is the SA12 configuration (RepMixer in stages 1–3, attention
in stage 4). T8, T12, and S12 use RepMixer in all four stages (see
`models/fastvit.py` in apple/ml-fastvit). The train/inference split
explains why the deployed artifacts are the already-reparameterized INT8
s12/sa12/t12/t8 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

Start with `runtime/python/main.py`; `classify.py` holds the model stages and `cli.py` handles options and results.
[runtime/python/README.md](runtime/python/README.md)

<a id="directory"></a>
## Directory structure

```text
fastvit/
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
| FastViT-T8 | 224x224 | ✓ | — | — | — |
| FastViT-T12 | 224x224 | ✓ | — | — | — |
| FastViT-S12 | 224x224 | ✓ | — | — | — |
| FastViT-SA12 | 224x224 | ✓ | — | — | — |

Runtime coverage: the Python CLI serves every published combination with
`--target auto|x5` and `--variant t8|t12|s12|sa12` (default `s12`). Only the
X5 archive publishes `FastViT_<size>_224x224_nv12.bin`; `--target s100`,
`s100p` and `s600` resolve no asset. The sample ships no C++ runtime.

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run` need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — output: "host dependencies: ok"
python3 -m venv .venv-fastvit
source .venv-fastvit/bin/activate
python3 -m pip install -r samples/vision/fastvit/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:fastvit:FastViT_S12_224x224_nv12.bin)
#    output: samples/vision/fastvit/model/FastViT_S12_224x224_nv12.bin
#    output: downloader exits 0 and prints the observed digest
bash samples/vision/fastvit/model/download.sh x5 s12

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    output: exit code 0 and a printed Top-5 list
python3 samples/vision/fastvit/runtime/python/main.py \
  --target x5 \
  --asset-id x5:fastvit:FastViT_S12_224x224_nv12.bin \
  --model-path samples/vision/fastvit/model/FastViT_S12_224x224_nv12.bin \
  --test-img samples/vision/fastvit/test_data/bucket.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`sa12`/`t12`/`t8` substitute their own reference and path (see
`--list-models`); the default variant (when none is given) is `s12`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0. Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `bucket.JPEG` the Top-5 contains a
bucket-related ImageNet class. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `bucket.JPEG`, with rank 1
being class 463 (bucket).

![Reference inference result on X5: bucket test image with the top-5
overlay, rank 1 class 463 (bucket)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance on RDK X5 (X5 release x5-v1.1.3; Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| FastViT-SA12 | 224x224 | 10.9 | 78.25% | 74.50% | 11.56 | 93.44 |
| FastViT-S12 | 224x224 | 8.8 | 76.50% | 72.00% | 5.86 | 193.87 |
| FastViT-T12 | 224x224 | 6.8 | 74.75% | 70.43% | 4.97 | 234.78 |
| FastViT-T8 | 224x224 | 3.6 | 73.50% | 68.50% | 2.09 | 667.21 |

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream FastViT distribution; upstream
model/weights licensing is governed by that distribution (see the paper
link above). Published artifacts follow the platform release manifests;
Python code is licensed under Apache-2.0. Review applicable upstream model and weight license terms for those components.

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
| `timm-t8-apple-in1k` | `fastvit_t8_apple_in1k_bayese_224x224_nv12.bin` | [`cbc6c66a2490`](https://huggingface.co/timm/fastvit_t8.apple_in1k/blob/cbc6c66a249060c66b8236284c7c3998e900e944/README.md) | not-run |
| `timm-t12-apple-in1k` | `fastvit_t12_apple_in1k_bayese_224x224_nv12.bin` | [`7630f7ea7919`](https://huggingface.co/timm/fastvit_t12.apple_in1k/blob/7630f7ea7919657fcfcf7f84808793b8c3b39e1d/README.md) | not-run |
| `timm-s12-apple-in1k` | `fastvit_s12_apple_in1k_bayese_224x224_nv12.bin` | [`662e9e267777`](https://huggingface.co/timm/fastvit_s12.apple_in1k/blob/662e9e2677774a03870672a03974c6a3cfe575ac/README.md) | not-run |
| `timm-sa12-apple-in1k` | `fastvit_sa12_apple_in1k_bayese_224x224_nv12.bin` | [`4f87441b01b5`](https://huggingface.co/timm/fastvit_sa12.apple_in1k/blob/4f87441b01b509bc759497692b7d342c2999923b/README.md) | not-run |

```bash
python3 samples/vision/fastvit/model/download.py --target x5 --variant timm-t8-apple-in1k
python3 samples/vision/fastvit/runtime/python/main.py --target x5 --variant timm-t8-apple-in1k --dry-run
```

Exact model URLs/SHA-256/size are in `docs/release/x5/models.yaml`; source and host
receipts are in `conversion/converted.json`. Each weight retains its own model
license in its public release manifest. No board result is implied by download.
<!-- /converted-batch-20261102 -->

<!-- converted-all-20261103 -->

## Converted S profiles (2026-11-03)

Independent converted profiles and explicit commands are listed in [model preparation](model/README.md). Board accuracy and performance are not measured yet.
