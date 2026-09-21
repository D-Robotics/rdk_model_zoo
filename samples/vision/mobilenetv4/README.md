# MobileNetV4 image classification

MobileNetV4 ImageNet-1k classification on RDK boards: one BGR image in, a
stable Top-K of `(class id, score, label)` out. Source model: [timm/models/MobileNetV4.py](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/MobileNetV4.py),
paper [MobileNetV4 -- Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518). [中文说明](README_cn.md)

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
| x5 | small | python | supported-verified (x5 8GB + 4GB board smoke, 2026-09-21) |
| x5 | medium | python | supported-verified (x5 8GB + 4GB board smoke, 2026-09-21) |
| s100 | small | python | supported-verified (S100 board smoke, 2026-09-21) |
| s100 | medium | python | supported-verified (S100 board smoke, 2026-09-21) |
| s600 | medium | python | supported-verified (S600 board smoke, 2026-09-21) |
| s600 | small | python | supported-verified (S600 board smoke, 2026-09-21) |
| s100p | any | python, cpp | not-supported (no s100p asset row in the release manifest; rejection verified on S100P hardware 2026-09-21 — explicit error, no fallback) |

Source baselines: X5 rdk_x5 @ac11571 (x5-v1.1.3); S rdk_s @380e1a2 (s-v1.1.2). The unified sample's host
tests all pass. Board smoke (2026-09-21) passed on x5 8GB/4GB and
S100/S600 with outputs identical across boards and equal to the source
implementations (the medium S artifact ran at its published 256x256
geometry); S100P was verified as a rejection case only. Evidence:
[B1 board smoke](../../../docs/releases/unified-migration/evidence/2026-09-21-b1-board-smoke-evidence.json).

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-mobilenetv4
source .venv-mobilenetv4/bin/activate
python3 -m pip install -r samples/vision/mobilenetv4/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin)
#    output: samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/mobilenetv4/model/download.sh x5 --variant small

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/mobilenetv4/runtime/python/main.py \
  --target x5 \
  --asset-id x5:mobilenetv4:MobileNetV4_conv_small_224x224_nv12.bin \
  --model-path samples/vision/mobilenetv4/model/MobileNetV4_conv_small_224x224_nv12.bin \
  --test-img samples/vision/mobilenetv4/test_data/great_grey_owl.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:` reference (see `--list-models`) and the
same root `datasets/imagenet/` labels. Full commands:
[runtime/python/README.md](runtime/python/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given. On X5 with the bundled `great_grey_owl.JPEG` the Top-1 matches the
image subject (a great grey owl); on S100/S600 with `zebra_cls.jpg` the
Top-5 includes `zebra`. A board that cannot be identified, or a target
without a matching artifact, exits with an error instead of guessing.

<a id="performance"></a>
## Performance data

Published MobileNetV4 performance on `RDK X5` from rdk_x5 @ac11571 (x5-v1.1.3):

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MobileNetV4-Conv-Medium | 224x224 | 1000 | 9.7 | 76.8% | 75.1% | 2.42 | 572+ |
| MobileNetV4-Conv-Small | 224x224 | 1000 | 3.8 | 70.8% | 68.8% | 1.18 | 1436+ |

The S-series source release (rdk_s @380e1a2 (s-v1.1.2)) published no latency or accuracy
figures for this model; none are inferred here.

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [conversion/](conversion/README.md) — conversion record and reference configurations
- [evaluator/](evaluator/README.md) — published benchmarks and functional checks
- `test_data/` — bundled test images ([great_grey_owl.JPEG](test_data/great_grey_owl.JPEG), [zebra_cls.jpg](test_data/zebra_cls.jpg))
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
source model is the upstream MobileNetV4 distribution; upstream model/weights
licensing is governed by that distribution (see the reference-implementation
link above). Published artifacts follow the platform release manifests; the
manifests carry no separate license field, and no additional license is
claimed here.
