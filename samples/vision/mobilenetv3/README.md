# MobileNetV3 image classification

MobileNetV3 ImageNet-1k classification on RDK boards: one BGR image in, a
stable Top-K of `(class id, score, label)` out. Source model: [timm/models/mobilenetv3.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/mobilenetv3.py),
paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). [中文说明](README_cn.md)

<a id="overview"></a>
## Overview

The maintained implementation is one Python flow (all targets). Python resolves one exact artifact reference from the platform
release manifests, verifies the board identity, loads `hbm_runtime`
lazily, and runs a `pre_process → forward → post_process` task
([runtime/python/README.md](runtime/python/README.md)).
The former platform branch entries remain compatibility shims under
`platforms/{x5,s}/` until the migration closeout; their audit record
lives in the migration documents, not here.

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | mobilenetv3 | python | supported-host-verified (B1 migration; board smoke pending) |
| s100 | mobilenetv3 | python | supported-host-verified (B1 migration; board smoke pending) |
| s600 | mobilenetv3 | python | supported-not-run (artifact published; board smoke pending) |
| s100p | any | python, cpp | not-supported (no s100p asset row in the release manifest) |

Source baselines: X5 rdk_x5 @ac11571 (x5-v1.1.3); S rdk_s @380e1a2 (s-v1.1.2). The unified sample's host
tests all pass; board smoke is executed by the user on X5 8GB/4GB and
S100, after which the status rows are updated.

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-mobilenetv3
source .venv-mobilenetv3/bin/activate
python3 -m pip install -r samples/vision/mobilenetv3/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:mobilenetv3:MobileNetV3_224x224_nv12.bin)
#    output: samples/vision/mobilenetv3/model/MobileNetV3_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv3/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv3/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv3:MobileNetV3_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv3/model/MobileNetV3_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv3/test_data/kit_fox.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given. On X5 with the bundled `kit_fox.JPEG` the Top-1 matches the
image subject (a kit fox); on S100/S600 with `zebra_cls.jpg` the
Top-5 includes `zebra`. A board that cannot be identified, or a target
without a matching artifact, exits with an error instead of guessing.

<a id="performance"></a>
## Performance data

Published MobileNetV3 performance on `RDK X5` from rdk_x5 @ac11571 (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV3-Large | 224x224 | 1000 | 5.5 | 74.8% | 64.8% | 2.02 | 714+ |

The S-series source release (rdk_s @380e1a2 (s-v1.1.2)) published no latency or accuracy
figures for this model; none are inferred here.

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — conversion record and reference configurations
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([kit_fox.JPEG](test_data/kit_fox.JPEG), [zebra_cls.jpg](test_data/zebra_cls.jpg))
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
source model is the upstream MobileNetV3 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
