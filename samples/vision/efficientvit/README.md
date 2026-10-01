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

The maintained implementation is one Python flow (X5 only; this sample has
no S-branch delivery and no C++ runtime on either source). Python resolves
one exact artifact reference from the platform release manifests, verifies
the board identity, loads `hbm_runtime` lazily, and runs a
`pre_process → forward → post_process` task
([runtime/python/README.md](runtime/python/README.md)).
The former platform branch entry remains a compatibility shim under
`platforms/x5/` until the migration closeout; its audit record lives in the
migration documents, not here.

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

Source-release feature summary (rdk_x5 @ac11571, x5-v1.1.3):

- **Memory-efficient attention**: reduces the data-movement overhead that limits standard transformer inference efficiency.
- **Cascaded group attention**: improves representation capability while controlling deployment cost.
- **Deployment-friendly normalization**: batch normalization simplifies inference-side fusion.
- **Classification output**: Top-K class IDs and confidence scores for ImageNet-1k labels.

![Runtime profiling](./test_data/comparison_between_transformer_and_cnn.png)

*Runtime profiling, restored from the X5 source release
(`test_data/comparison_between_transformer_and_cnn.png`, rdk_x5 @ac11571,
sha256 `be1e2e39…`; Figure 2 of the paper): memory-bound operations
(red labels) take a large share of Swin-T/DeiT-T latency — the overhead
EfficientViT targets.*

![MHSA proportion study](./test_data/mhsa_computation.jpg)

*(`test_data/mhsa_computation.jpg`, rdk_x5 @ac11571, sha256
`4dda6352…`; Figure 3 of the paper): top-1 accuracy of downscaled
Swin-T/DeiT-T baselines against the proportion of MHSA layers — thinning
MHSA alone does not by itself produce an efficient design, motivating
the cascaded group attention instead.*

![EfficientViT architecture](./test_data/efficientvit_msra_architecture.png)

*EfficientViT overview, restored from the X5 source release
(`test_data/efficientvit_msra_architecture.png`, rdk_x5 @ac11571, sha256
`403d1c63…`; Figure 6 of the paper): (a) the three-stage network with
overlap patch embedding, (b) the sandwich-layout block, and (c) cascaded
group attention with per-head cascading and concat projection.*

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | m5 | python | supported (board smoke passed 2026-09-21 on x5-8g + x5-4g, see below) |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no EfficientViT asset; selection is an explicit error, no cross-platform fallback) |

Source baseline: X5 rdk_x5 @ac11571 (x5-v1.1.3). The unified sample's host
tests (26) all pass. Board smoke (2026-09-21; same board, same artifact
bytes, same input image, legacy wrapper vs unified entry): x5-8g and
x5-4g return exactly equal m5 top-5 ids (max abs score diff <=1.9e-9);
the `run.sh` CLI exited 0 on both boards. Raw-tensor equality, dataset
accuracy, and latency are not covered; the published benchmark tables
remain source records. Evidence: [B2 board smoke](../../../docs/releases/unified-migration/evidence/2026-09-21-b2-board-smoke-evidence.json).

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

The default variant (when none is given) is `m5` — the only published
variant, preserving the source entrypoint's default model. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given (the legacy entrypoint always wrote `test_data/result.jpg` — that
side effect is gone). With the bundled `hook.JPEG` the Top-5 contains
hook-related ImageNet classes. A board that cannot be identified, or a
target without a matching artifact (all S targets), exits with an error
instead of guessing.

<a id="performance"></a>
## Performance data

Published records from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3), not re-measured in this repository (source notes: Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model; the
source table does not state the latency threading conditions):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EfficientViT_m5 | 224x224 | 1000 | 12.4 | 73.75% | 72.50% | 6.34 | 174.70 |

![Inference result](./test_data/inference.png)

*Historical inference screenshot from the X5 source release
(rdk_x5 @ac11571, `test_data/inference.png`, sha256 `2a23e138…`): the
bundled [hook.JPEG](test_data/hook.JPEG) ranks `hook` first, followed by
crane, chain, seashore, and dock. Recorded by the source release on its
own runtime entry — not a new run of this repository.*

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — X5 reference PTQ config with disclosed gaps
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
Published artifacts follow the platform release manifests; the manifests
carry no separate license field, and no additional license is claimed here.
