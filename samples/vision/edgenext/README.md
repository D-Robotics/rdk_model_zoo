# EdgeNeXt image classification

EdgeNeXt ImageNet-1k classification on RDK X5: one BGR image in, a
stable Top-K of `(class id, score, label)` out. The X5 release ships the
base, small, x-small, and xx-small variants (paper [EdgeNeXt: Efficiently
Amalgamated CNN-Transformer Architecture for Mobile Vision
Applications](https://arxiv.org/abs/2206.10589), reference
[mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt)).
[中文说明](README_cn.md)

<a id="overview"></a>

## Overview

EdgeNeXt is an efficient hybrid CNN-Transformer architecture for mobile
vision: a four-stage pyramid combines convolutional encoders with SDTA
(Split Depth-wise Transpose Attention) encoders to balance classification
accuracy, model size, and inference speed. It targets ImageNet-1k
1000-class classification. The source README's four feature highlights:

- **Hybrid CNN-Transformer design** — combines the inference efficiency
  of convolutions with transformer-style global feature modeling.
- **Four-stage pyramid** — a deployment-friendly hierarchical feature
  extraction structure.
- **SDTA encoder** — encodes multi-scale features through channel
  splitting (split 3×3 branches) and transpose attention.
- **Efficient deployment** — base, small, x-small, and xx-small RDK X5
  deployment models with packed NV12 input.

![EdgeNeXt architecture: four-stage pyramid with NxN convolution encoder
and SDTA encoder details](./test_data/EdgeNeXt_architecture.png)

*Figure: the EdgeNeXt architecture — the four-stage pyramid (top) with
the NxN convolution encoder (bottom left) and the SDTA encoder with its
split 3×3 branches and transpose attention (bottom right). Restored from
the X5 source README (rdk_x5 @ac11571, x5-v1.1.3); it depicts the
upstream training architecture, while the deployed artifacts are the
INT8-quantized base/small/x_small/xx_small variants at 224×224 NV12 (see
[Support matrix](#support-matrix)).*

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
| x5 | base, small, x_small, xx_small | python | supported (source-verified contract; board smoke pending for B3, see below) |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no EdgeNeXt asset; selection is an explicit error, no cross-platform fallback) |

Source baseline: X5 rdk_x5 @ac11571 (x5-v1.1.3). The unified sample's host
tests (26) all pass. Board smoke for this batch (B3) is executed after
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
python3 -m venv .venv-edgenext
source .venv-edgenext/bin/activate
python3 -m pip install -r samples/vision/edgenext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:edgenext:EdgeNeXt_base_224x224_nv12.bin)
#    output: samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/edgenext/model/download.sh x5 base

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/edgenext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:edgenext:EdgeNeXt_base_224x224_nv12.bin \
  --model-path samples/vision/edgenext/model/EdgeNeXt_base_224x224_nv12.bin \
  --test-img samples/vision/edgenext/test_data/Zebra.jpg \
  --label-file datasets/imagenet/imagenet_classes.names
```

`small`/`x_small`/`xx_small` substitute their own reference and path
(see `--list-models`); the default variant (when none is given) is
`base`, preserving the source
entrypoint's default model. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given (the legacy entrypoint always wrote `test_data/result.jpg` — that
side effect is gone). With the bundled `Zebra.jpg` the Top-5 contains a
zebra-related ImageNet class. A board that cannot be identified, or a
target without a matching artifact (all S targets), exits with an error
instead of guessing.

For reference, the X5 source README (rdk_x5 @ac11571, x5-v1.1.3 legacy
Python entrypoint) illustrated its run with the screenshot below: the
legacy `result.jpg` drawing overlaid the top-5 ranks on the image, with
rank 1 being class 340 (zebra). This is a historical screenshot from the
source delivery, not a run of the current entrypoint in this repository.

![Historical inference screenshot from the X5 source README (rdk_x5
@ac11571): zebra test image with the legacy top-5 overlay, rank 1 class
340 (zebra)](./test_data/inference.png)

<a id="performance"></a>
## Performance data

Published records from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3), not re-measured in this repository (source notes: Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model, latency
single-frame single-thread single-core, FPS multi-threaded):

| Model | Size | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- |
| EdgeNeXt-base | 224x224 | 18.51 | 78.21% | 74.52% | 8.80 | 113.35 |
| EdgeNeXt-small | 224x224 | 5.59 | 76.50% | 71.75% | 4.41 | 226.15 |
| EdgeNeXt-x-small | 224x224 | 2.34 | 71.75% | 66.25% | 2.88 | 345.73 |
| EdgeNeXt-xx-small | 224x224 | 1.33 | 69.50% | 64.25% | 2.47 | 403.49 |

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — X5 reference PTQ configs with disclosed gaps
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([Zebra.jpg](test_data/Zebra.jpg) plus reference illustrations)
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
source models are the upstream EdgeNeXt distribution; upstream
model/weights licensing is governed by that distribution (see the paper
link above). Published artifacts follow the platform release manifests;
the manifests carry no separate license field, and no additional license
is claimed here.
