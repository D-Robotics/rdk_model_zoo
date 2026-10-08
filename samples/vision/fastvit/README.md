# FastViT image classification

FastViT ImageNet-1k classification on RDK X5: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 release ships the
S12, SA12, T12, and T8 variants (paper [FastViT: A Fast Hybrid Vision
Transformer using Structural
Reparameterization](https://arxiv.org/abs/2303.14189)).
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

The sample provides a Python runtime for X5. The
`FastViTClassifier` class runs a `preprocess → infer → postprocess` flow
chained by `predict`: it resolves one exact artifact reference from the
platform release manifest, verifies the board identity, loads
`hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

<a id="directory"></a>
## Directory structure

```text
fastvit/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
└── requirements-host.txt  # Source or data file
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | s12, sa12, t12, t8 | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no FastViT asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
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
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/fastvit/model/download.sh x5 s12

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
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
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `bucket.JPEG` the Top-5 contains a
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
