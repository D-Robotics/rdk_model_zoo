# FasterNet image classification

FasterNet ImageNet-1k classification on RDK X5: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 release ships the
S, T0, T1, and T2 variants (paper [Run, Don't Walk: Chasing Higher FLOPS
for Faster Neural Networks](https://arxiv.org/abs/2303.03667), as cited by
the source delivery).
[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

FasterNet is a lightweight CNN family designed around one idea: chase a
higher *effective* FLOPS (actual computed operations per second) instead
of only minimizing theoretical FLOPs. Its core operator, partial
convolution (PConv), applies the spatial convolution to only a fraction
of the input channels and leaves the rest untouched, cutting redundant
memory access and improving practical runtime efficiency on edge
devices. It targets ImageNet-1k 1000-class classification. The source
README's four feature highlights:

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
is faster at the same FLOPs. Restored from the X5 source README
(rdk_x5 @ac11571, x5-v1.1.3); conditions are the upstream CPU
measurements, not RDK board numbers (see [Performance
data](#performance)).*

![FasterNet architecture: four-stage hierarchy and the FasterNet block
with partial convolution](./test_data/FasterNet_architecture.png)

*Figure (upstream paper Fig. 4): the overall architecture — four
hierarchical stages of FasterNet blocks with embedding/merging layers —
the PConv detail (convolution applied only to a channel fraction) and
the block layout PConv 3×3 → two pointwise convs, with normalization and
activation only after the middle layer to preserve feature diversity.
Restored from the X5 source README (rdk_x5 @ac11571); it depicts the
upstream training architecture, while the deployed artifacts are the
INT8-quantized s/t0/t1/t2 variants at 224×224 NV12 (see [Support
matrix](#support-matrix)).*

The maintained implementation is one Python flow (X5 only; this sample has
no S-branch delivery and no C++ runtime on either source). Python resolves
one exact artifact reference from the platform release manifests, verifies
the board identity, loads `hbm_runtime` lazily, and runs a
`pre_process → forward → post_process` task
([runtime/python/README.md](runtime/python/README.md)).
The former platform branch entry remains a compatibility shim under
`platforms/x5/` until the migration closeout; its audit record lives in the
migration documents, not here.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | s, t0, t1, t2 | python | supported (source-verified contract; board smoke pending for B3, see below) |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no FasterNet asset; selection is an explicit error, no cross-platform fallback) |

Source baseline: X5 rdk_x5 @ac11571 (x5-v1.1.3). The unified sample's host
tests (28) all pass. Board smoke for this batch (B3) is executed after
the host side of all four B3 samples lands; this matrix is updated with
the observed results then — until that entry exists, board status for this
sample is **not-run**, and the legacy source remains the verified delivery.

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
`--list-models`); the default variant (when none is given) is `s`,
preserving the source
entrypoint's default model. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given (the legacy entrypoint always wrote `test_data/result.jpg` — that
side effect is gone). With the bundled `drake.JPEG` the Top-5 contains a
drake-related ImageNet class. A board that cannot be identified, or a
target without a matching artifact (all S targets), exits with an error
instead of guessing.

For reference, the X5 source README (rdk_x5 @ac11571, x5-v1.1.3 legacy
Python entrypoint) illustrated its run with the screenshot below: the
legacy `result.jpg` drawing overlaid the top-5 ranks on the image, with
rank 1 being class 97 (drake). This is a historical screenshot from the
source delivery, not a run of the current entrypoint in this repository.

![Historical inference screenshot from the X5 source README (rdk_x5
@ac11571): drake test image with the legacy top-5 overlay, rank 1 class
97 (drake)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published records from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3), not re-measured in this repository (source notes: Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| FasterNet-S | 224x224 | 31.1 | 77.04% | 76.15% | 6.73 | 162.83 |
| FasterNet-T2 | 224x224 | 15.0 | 76.50% | 76.05% | 3.39 | 342.48 |
| FasterNet-T1 | 224x224 | 7.6 | 74.29% | 71.25% | 1.96 | 708.40 |
| FasterNet-T0 | 224x224 | 3.9 | 71.75% | 68.50% | 1.41 | 1135.13 |

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — X5 reference PTQ configs with disclosed gaps
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([drake.JPEG](test_data/drake.JPEG) plus reference illustrations)
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
source models are the upstream FasterNet distribution; upstream
model/weights licensing is governed by that distribution (see the paper
link above). Published artifacts follow the platform release manifests;
the manifests carry no separate license field, and no additional license
is claimed here.
