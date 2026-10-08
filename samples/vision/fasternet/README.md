English | [简体中文](README_cn.md)

# FasterNet image classification

FasterNet uses partial convolution to reduce computation and memory access in image classification.

Sources: [Run, Don't Walk: Chasing Higher FLOPS
for Faster Neural Networks](https://arxiv.org/abs/2303.03667)

[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

FasterNet is a lightweight CNN family designed around one idea: chase a
higher *effective* FLOPS (actual computed operations per second) instead
of only minimizing theoretical FLOPs. Its core operator, partial
convolution (PConv), applies the spatial convolution to only a fraction
of the input channels and leaves the rest untouched, cutting redundant
memory access and improving practical runtime efficiency on edge
devices. It targets ImageNet-1k 1000-class classification. Four feature
highlights:

- **High-FLOPS design** — emphasizes practical compute efficiency
  instead of minimizing theoretical FLOPs only.
- **Partial convolution (PConv)** — reduces redundant computation and
  memory access.
- **Lightweight CNN backbone** — keeps a deployment-friendly CNN
  structure for efficient board-side inference.
- **Efficient deployment** — S, T0, T1, and T2 RDK X5 deployment models
  with packed NV12 input.

![Effective FLOPS and latency comparison against other networks on
CPU](./test_data/FLOPs%20of%20Nets.png)

*Figure (upstream paper Fig. 2): (a) FLOPS under varied FLOPs on CPU —
many networks run at a lower effective FLOPS than ResNet50, while
FasterNet stays high; (b) latency under varied FLOPs on CPU — FasterNet
is faster at the same FLOPs. The measurements are the upstream CPU
source-paper benchmarks; RDK board numbers appear in [Performance
data](#performance)).*

![FasterNet architecture: four-stage hierarchy and the FasterNet block
with partial convolution](./test_data/FasterNet_architecture.png)

*Figure (upstream paper Fig. 4): the overall architecture — four
hierarchical stages of FasterNet blocks with embedding/merging layers —
the PConv detail (convolution applied only to a channel fraction) and
the block layout PConv 3×3 → two pointwise convs, with normalization and
activation only after the middle layer to preserve feature diversity.
The figure shows the upstream training architecture; the deployed
artifacts are the INT8-quantized s/t0/t1/t2 variants at 224×224 NV12
(see [Support matrix](#support-matrix)).*

Start with `runtime/python/main.py`; `classify.py` holds the model stages and `cli.py` handles options and results.
[runtime/python/README.md](runtime/python/README.md)

<a id="directory"></a>
## Directory structure

```text
fasternet/
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

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | s, t0, t1, t2 | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no FasterNet asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-fasternet
source .venv-fasternet/bin/activate
python3 -m pip install -r samples/vision/fasternet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:fasternet:FasterNet_S_224x224_nv12.bin)
#    output: samples/vision/fasternet/model/FasterNet_S_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/fasternet/model/download.sh x5 s

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/fasternet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:fasternet:FasterNet_S_224x224_nv12.bin \
  --model-path samples/vision/fasternet/model/FasterNet_S_224x224_nv12.bin \
  --test-img samples/vision/fasternet/test_data/drake.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

`t0`/`t1`/`t2` substitute their own reference and path (see
`--list-models`); the default variant (when none is given) is `s`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `drake.JPEG` the Top-5 contains a
drake-related ImageNet class. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

The screenshot below shows a reference run from the X5 release: the demo
overlay draws the top-5 ranks onto the bundled `drake.JPEG`, with rank 1
being class 97 (drake).

![Reference inference result on X5: drake test image with the top-5
overlay, rank 1 class 97 (drake)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published performance on RDK X5 (X5 release x5-v1.1.3; Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 1135.13 |

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream FasterNet distribution; upstream
model/weights licensing is governed by that distribution (see the paper
link above). Published artifacts follow the platform release manifests;
Python code is licensed under Apache-2.0. Review applicable upstream model and weight license terms for those components.
