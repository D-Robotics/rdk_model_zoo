# EfficientViT image classification

EfficientViT (MSRA cascaded-group-attention series) ImageNet-1k
classification on RDK X5: one BGR image in, a stable Top-K of
`(class id, score, label)` out. The X5 release ships the m5 variant
(paper [EfficientViT: Memory Efficient Vision Transformer with Cascaded
Group Attention](https://arxiv.org/abs/2305.07027), reference
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)).
[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

The sample provides a Python runtime for X5. The `EfficientViTClassifier` class runs a `preprocess → infer → postprocess` flow chained by `predict`: it resolves one exact artifact reference from the platform release manifest, verifies the board identity, loads `hbm_runtime` lazily, and returns a typed Top-K result
([runtime/python/README.md](runtime/python/README.md)).

### Algorithm background

EfficientViT (MSRA) attacks the memory-bound cost of standard
self-attention: runtime profiling shows that reshape/normalization/copy
data movement takes a large share of Swin/DeiT latency (paper Figure 2),
and simply thinning out MHSA layers hurts accuracy (paper Figure 3).
EfficientViT instead uses cascaded group attention — each attention head
receives the cascaded output of the previous heads, reducing the
per-head attention cost while improving representation — and batch
normalization for inference-friendly fusion
([paper](https://arxiv.org/abs/2305.07027),
[microsoft/Cream/EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)).

Source-release feature summary (x5-v1.1.3):

- **Memory-efficient attention**: reduces the data-movement overhead that limits standard transformer inference efficiency.
- **Cascaded group attention**: improves representation capability while controlling deployment cost.
- **Deployment-friendly normalization**: batch normalization simplifies inference-side fusion.
- **Classification output**: Top-K class IDs and confidence scores for ImageNet-1k labels.

![Runtime profiling](./test_data/comparison_between_transformer_and_cnn.png)

*Runtime profiling (Figure 2 of the paper): memory-bound operations
(red labels) take a large share of Swin-T/DeiT-T latency — the overhead
EfficientViT targets.*

![MHSA proportion study](./test_data/mhsa_computation.jpg)

*MHSA proportion study (Figure 3 of the paper): top-1 accuracy of downscaled
Swin-T/DeiT-T baselines against the proportion of MHSA layers — thinning
MHSA alone does not by itself produce an efficient design, motivating
the cascaded group attention instead.*

![EfficientViT architecture](./test_data/efficientvit_msra_architecture.png)

*EfficientViT overview (Figure 6 of the paper): (a) the three-stage
network with overlap patch embedding, (b) the sandwich-layout block, and
(c) cascaded group attention with per-head cascading and concat
projection.*

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | m5 | python | supported |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no EfficientViT asset; selection is an explicit error, no cross-platform fallback) |

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-efficientvit
source .venv-efficientvit/bin/activate
python3 -m pip install -r samples/vision/efficientvit/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:efficientvit:EfficientViT_m5_224x224_nv12.bin)
#    output: samples/vision/efficientvit/model/EfficientViT_m5_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/efficientvit/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/efficientvit/runtime/python/main.py \
  --target x5 \
  --asset-id x5:efficientvit:EfficientViT_m5_224x224_nv12.bin \
  --model-path samples/vision/efficientvit/model/EfficientViT_m5_224x224_nv12.bin \
  --test-img samples/vision/efficientvit/test_data/hook.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

The default variant is `m5`. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; Pass `--img-save-path` to save a visualization; otherwise results are printed to stdout. With the bundled `hook.JPEG` the Top-5 contains
hook-related ImageNet classes. Select a target and variant listed in the [Support matrix](#support-matrix), prepare that exact manifest artifact with the model downloader, and run the sample on the matching board.

<a id="performance"></a>
## Performance data

Published performance on RDK X5 (X5 release x5-v1.1.3; Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model; the
source table does not state the latency threading conditions):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

![Inference result](./test_data/inference.png)

*Reference inference result from the X5 release: the bundled
[hook.JPEG](test_data/hook.JPEG) ranks `hook` first, followed by crane,
chain, seashore, and dock.*

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — X5 PTQ configuration and model-specific preparation steps
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([hook.JPEG](test_data/hook.JPEG) plus reference illustrations)
- `tests/` — host unittest suite

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source models are the upstream MSRA EfficientViT distribution
([microsoft/Cream](https://github.com/microsoft/Cream/tree/main/EfficientViT));
upstream model/weights licensing is governed by that distribution.
Published artifact use follows the applicable platform release terms.
