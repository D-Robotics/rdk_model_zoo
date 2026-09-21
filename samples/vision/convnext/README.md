# ConvNeXt image classification

ConvNeXt (modernized ConvNet line) ImageNet-1k classification on RDK X5:
one BGR image in, a stable Top-K of `(class id, score, label)` out. The X5
release publishes the atto variant (paper [A ConvNet for the
2020s](https://arxiv.org/abs/2201.03545), reference
[facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt));
the source conversion directory additionally ships femto/nano PTQ recipes
with no published asset (see [conversion/README.md](conversion/README.md)).
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

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | atto | python | supported (source-verified contract; board smoke pending for B3, see below) |
| s100 / s100p / s600 | any | python | not-supported (the S manifest publishes no ConvNeXt asset; selection is an explicit error, no cross-platform fallback) |

Source baseline: X5 rdk_x5 @ac11571 (x5-v1.1.3). The unified sample's host
tests (28) all pass. Board smoke for this batch (B3) is executed after the
host side of all four B3 samples lands; this matrix is updated with the
observed results then — until that entry exists, board status for this
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
python3 -m venv .venv-convnext
source .venv-convnext/bin/activate
python3 -m pip install -r samples/vision/convnext/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:convnext:ConvNeXt_atto_224x224_nv12.bin)
#    output: samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/convnext/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/convnext/runtime/python/main.py \
  --target x5 \
  --asset-id x5:convnext:ConvNeXt_atto_224x224_nv12.bin \
  --model-path samples/vision/convnext/model/ConvNeXt_atto_224x224_nv12.bin \
  --test-img samples/vision/convnext/test_data/cheetah.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

The default variant (when none is given) is `atto` — the only published
variant, preserving the source entrypoint's default model. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given (the legacy entrypoint always wrote `test_data/result.jpg` — that
side effect is gone). With the bundled `cheetah.JPEG` the Top-5 contains
cheetah-related ImageNet classes. A board that cannot be identified, or a
target without a matching artifact (all S targets), exits with an error
instead of guessing.

<a id="performance"></a>
## Performance data

Published records from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3), not re-measured in this repository (source notes: Float Top-1
on the pre-quantization ONNX, Quant Top-1 on the deployment model; latency
single-frame single-thread single-core, FPS 4-thread concurrent; CPU
8xA55@1.8GHz performance mode, BPU 1xBayes-e@1GHz):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConvNeXt_nano | 224x224 | 1000 | 15.59 | 77.37% | 71.75% | 5.71 | 200+ |
| ConvNeXt_pico | 224x224 | 1000 | 9.04 | 77.25% | 71.03% | 3.37 | 364+ |
| ConvNeXt_femto | 224x224 | 1000 | 5.22 | 73.75% | 72.25% | 2.46 | 556+ |

The published table covers nano/pico/femto, not atto — the only variant
with a downloadable artifact has no benchmark row; recorded as published.

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — X5 reference PTQ config with disclosed gaps
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([cheetah.JPEG](test_data/cheetah.JPEG) plus reference illustrations)
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
source models are the upstream ConvNeXt distribution
([facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt));
upstream model/weights licensing is governed by that distribution.
Published artifacts follow the platform release manifests; the manifests
carry no separate license field, and no additional license is claimed here.
