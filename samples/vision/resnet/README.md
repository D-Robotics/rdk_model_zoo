# ResNet18 image classification

ResNet18 ImageNet-1k classification on RDK boards: one BGR image in, a
stable Top-K of `(class id, score, label)` out. Source model: TorchVision
ResNet18 ([upstream](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html));
this sample is the repository's single-model classification reference.
[中文说明](README_cn.md)

<a id="overview"></a>
## Overview

The maintained implementation is one Python flow (all targets) plus one
S-series C++ flow. Python resolves one exact artifact reference from the
platform release manifests, verifies the board identity, loads `hbm_runtime`
lazily, and runs a `pre_process → forward → post_process` task
([runtime/python/README.md](runtime/python/README.md)). The C++ flow keeps
the audited S-series `hbDNNInferV2` implementation
([runtime/cpp/README.md](runtime/cpp/README.md)). The former X5 and S18
Python entrypoints remain usable compatibility shims that call the same
canonical implementation; their audit record lives in the migration
documents, not here.

### Algorithm background

ResNet was proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian
Sun. Residual learning with shortcut connections lets each block learn a
residual instead of the full mapping, which stabilizes deep-network
optimization and avoids the degradation that plain layer stacks suffer as
depth grows ([paper](https://arxiv.org/abs/1512.03385),
[torchvision.models.resnet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)).

Variant notes carried over from the source deliveries (X5 rdk_x5 @ac11571;
S rdk_s @380e1a2 resnet18/resnet50/resnet152, merged here in B1):

- **resnet18** — the lightweight residual variant; the S delivery positions it for quick classification validation.
- **resnet50** — bottleneck residual blocks (`1x1 → 3x3 → 1x1`) build a deeper network with controlled computation.
- **resnet152** — the 152-layer design trades more compute for representational capacity.

![ResNet residual blocks](./test_data/ResNet_architecture.png)

*Restored from the source deliveries (the same file ships as
`test_data/ResNet_architecture.png` in rdk_x5 @ac11571 and as
`test_data/resnet_architecture.png` in rdk_s @380e1a2, sha256
`cebea796…`): the residual building block of ResNet-18/34 (left, two 3×3
convolutions) and the bottleneck building block of ResNet-50/101/152
(right, 1×1 → 3×3 → 1×1), Figure 5 of the ResNet paper.*

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Status |
| --- | --- | --- | --- |
| x5 | resnet18 | python | supported-verified (both X5 boards, 2026-09-17) |
| s100 | resnet18 | python | supported-verified (S100, 2026-09-17) |
| s100 | resnet18 | cpp | supported-verified (S100 build + run, Top-5 equal to source baseline, 2026-09-17) |
| s600 | resnet18 | python | supported-not-run (outside the B1 smoke set; board access recovered 2026-09-21) |
| s600 | resnet18 | cpp | supported-not-run (outside the B1 smoke set; board access recovered 2026-09-21) |
| s100 | resnet50 | python | supported-verified (S100 board smoke, 2026-09-21) |
| s600 | resnet50 | python | supported-verified (S600 board smoke, 2026-09-21) |
| s100 | resnet152 | python | supported-verified (S100 board smoke, 2026-09-21) |
| s600 | resnet152 | python | supported-verified (S600 board smoke, 2026-09-21) |
| s100p | any | python, cpp | not-supported (no ResNet asset row in the release manifest) |

Verification evidence: [integration review 2026-09-17](../../../docs/releases/unified-migration/2026-09-17-integration-review.md).
ResNet50/152 joined this sample in B1 from the S branch (no X5 artifact is
published for them); they resolve the `s:resnet50`/`s:resnet152` manifest
rows and run the same flow with `--variant resnet50`/`--variant resnet152`.
ResNet50/152 board smoke passed on S100 and S600 (2026-09-21): Top-1
zebra with score tensors equal to the source implementations (maxdiff
≤1.2e-7). Evidence: [B1 board smoke](../../../docs/releases/unified-migration/evidence/2026-09-21-b1-board-smoke-evidence.json).

<a id="prerequisites"></a>
## Prerequisites

Board Python execution needs the board image's matching `hbm_runtime`,
NumPy, and OpenCV-Python; the SDK is imported only when a model executes
(`--help`, `--list-models`, `--dry-run`, and host tests need no SDK).
Manifest reading also needs PyYAML. On a development host, install the
user-space dependencies from `requirements-host.txt`:

```bash
# cwd: repository root — success: "host dependencies: ok"
python3 -m venv .venv-resnet
source .venv-resnet/bin/activate
python3 -m pip install -r samples/vision/resnet/requirements-host.txt
python3 -c "import cv2, numpy, yaml; print('host dependencies: ok')"
```

The C++ build needs CMake, a C++17 compiler, OpenCV/gflags/fmt development
packages, and the Horizon DNN headers/libraries — see
[runtime/cpp/README.md](runtime/cpp/README.md). Model conversion runs in
the x86 OpenExplore environment, not on the board — see
[conversion/README.md](conversion/README.md).

<a id="quickstart"></a>
## Quick start

One complete path on an X5 board, commands run from the repository root.
Prerequisite: the board image with `hbm_runtime` and network access to the
manifest's model server.

```bash
# 1. Prepare the artifact (input: manifest row x5:resnet:resnet18_224x224_nv12.bin)
#    output: samples/vision/resnet/model/resnet18_224x224_nv12.bin
#    success: downloader exits 0 and prints the observed digest
bash samples/vision/resnet/model/download.sh x5

# 2. Run classification (input: the artifact above plus the bundled test image)
#    output: Top-5 class ids, scores, labels on stdout
#    success: exit code 0 and a printed Top-5 list
python3 samples/vision/resnet/runtime/python/main.py \
  --target x5 \
  --asset-id x5:resnet:resnet18_224x224_nv12.bin \
  --model-path samples/vision/resnet/model/resnet18_224x224_nv12.bin \
  --test-img samples/vision/resnet/test_data/white_wolf.JPEG \
  --label-file datasets/imagenet/imagenet_classes.names
```

For S100/S600 use the matching `s:resnet18:<target>/...` reference and the
the same root `datasets/imagenet/` labels; for the C++ flow use
`bash samples/vision/resnet/runtime/cpp/run.sh`. Full commands:
[runtime/python/README.md](runtime/python/README.md),
[runtime/cpp/README.md](runtime/cpp/README.md).

<a id="expected-results"></a>
## Expected results

The Python run prints a stable Top-K (default 5) of class IDs, scores, and
labels and exits 0; no output files are written unless `--img-save-path` is
given. The 2026-09-17 board comparison recorded the canonical run matching
the legacy entrypoints on both X5 boards and S100 (same class IDs and raw
scores for the same artifact, image, and resize mode) — see the integration
review linked above. The C++ binary prints the Top-K lines with the label
file's names. A board that cannot be identified, or a target without a
matching artifact, exits with an error instead of guessing.

Historical result screenshots restored from the S source delivery
(rdk_s @380e1a2, one delivery per variant; the source does not state which
board produced them): each shows the legacy S-runtime result for the
bundled [zebra_cls.jpg](test_data/zebra_cls.jpg) — zebra, class ID 340 —
with the recorded top-1 confidence 0.9985 (resnet18), 0.9956 (resnet50),
and 0.9649 (resnet152). These captures document the source deliveries,
not a run of the unified sample.

![ResNet18 source result](./test_data/result_resnet18_s.png)
![ResNet50 source result](./test_data/result_resnet50_s.png)
![ResNet152 source result](./test_data/result_resnet152_s.png)

<a id="performance"></a>
## Performance data

Published ResNet18 record from the X5 source release (rdk_x5 @ac11571,
x5-v1.1.3), not re-measured in this repository. The source does not state
the latency/FPS threading conditions; the same figure is preserved in the
[evaluator record](evaluator/README.md). The S source release (rdk_s
@380e1a2) published no latency or accuracy figures for ResNet18/50/152;
none are inferred here.

| Model | Size | Classes | Params (M) | Float Top-1 | Quant Top-1 | Latency (ms) | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 224x224 | 1000 | 11.2 | 71.5% | 70.5% | 2.95 | 449+ |

![Inference result](./test_data/inference.png)

*Historical inference screenshot from the X5 source release
(rdk_x5 @ac11571, `test_data/inference.png`, sha256 `16c9d04e…`): the
bundled [white_wolf.JPEG](test_data/white_wolf.JPEG) ranks `white wolf`
first, followed by Arctic fox, timber wolf, Samoyed, and polar bear.
Recorded by the source release on its own runtime entry — not a new run
of this repository.*

<a id="directory"></a>
## Directory

- [model/](model/README.md) — manifest-driven artifact download, no checked-in binaries
- [runtime/python/](runtime/python/README.md) — canonical Python entrypoint and task modules
- [runtime/cpp/](runtime/cpp/README.md) — S-series C++ source, CMake, launcher
- [conversion/](conversion/README.md) — ONNX export and OE conversion record
- [evaluator/](evaluator/README.md) — host checks and functional board checks
- `test_data/` — bundled test images ([white_wolf.JPEG](test_data/white_wolf.JPEG), [zebra_cls.jpg](test_data/zebra_cls.jpg))
- `tests/` — host unittest suite

<a id="entry-points"></a>
## Entry points

- Model preparation: [model/README.md](model/README.md)
- Python runtime: [runtime/python/README.md](runtime/python/README.md)
- C++ runtime (S100/S600): [runtime/cpp/README.md](runtime/cpp/README.md)
- Conversion: [conversion/README.md](conversion/README.md)
- Evaluation: [evaluator/README.md](evaluator/README.md)

<a id="license"></a>
## License

Sample code follows the repository top-level LICENSE (Apache-2.0). The
source model is TorchVision ResNet18; upstream model/weights licensing is
governed by the TorchVision distribution (see the upstream link above).
Published artifacts follow the platform release manifests; the manifests
carry no separate license field, and no additional license is claimed here.
