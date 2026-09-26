[English](README.md) | [简体中文](README_cn.md)

# YOLO26 Depth

<a id="overview"></a>
## Overview

Estimate a dense relative-depth map from one BGR image, preserving X5 and S
source capabilities in one sample. It includes Python inference on four targets,
X5 C++, explicit model preparation, both conversion toolchains and offline depth
evaluation. Rendering, file IO, model binding and SDK ownership are separate from
the three inference stages. Human users and Agents use the same native commands.

The published profile is target-dependent. All X5 variants and S n/s/m accept
letterboxed NV12 and return calibrated log-depth; S l/x accept normalized RGB
featuremaps and return raw logits. Only the latter need CPU clip/scale/bias.
Both then produce original-size **relative depth**, not calibrated metres.

<a id="support-matrix"></a>
## Support matrix

| Target | march | n/s/m | l/x | Runtime |
|---|---|---|---|---|
| X5 | bayes-e | NV12 / calibrated log | NV12 / calibrated log | Python + C++ |
| S100 | nash-e | NV12 / calibrated log | featuremap / raw logit | Python |
| S100P | nash-m | NV12 / calibrated log | featuremap / raw logit | Python |
| S600 | nash-p | NV12 / calibrated log | featuremap / raw logit | Python |

Default variant is n on every target. Twenty manifest assets exist, including
five distinct S100P assets. Source availability is not new runtime acceptance:
this migration has host tests, while board inference, real native SDK/OpenCV
build, Torch export, OE compilation and dataset measurements remain not-run.
S native depth is not claimed. Historical source results appear below and in
[evaluator](evaluator/README.md), with their unresolved evidence boundaries.

<a id="prerequisites"></a>
## Prerequisites

Use the corresponding board Linux image and vendor `hbm_runtime` for Python
inference, plus Python, NumPy, OpenCV and PyYAML. Host list/dry-run requires no
board SDK. Model files are prepared explicitly; inference does not download or
install anything. Runtime checks concrete board identity before loading a model.
X5 C++ dependencies and build steps are in [native runtime](runtime/cpp/README.md).
Conversion runs in a separately prepared x86 toolchain environment.

Commands below run from repository root. Download needs network access; inference
uses local artifacts. See [model preparation](model/README.md) for hashes and
external paths. The example image is [bus.jpg](test_data/bus.jpg), not a dataset.

<a id="quickstart"></a>
## Quick start

Inspect selections on any host:

```bash
python -m samples.vision.yolo26_depth.runtime.python.main --list-models
python -m samples.vision.yolo26_depth.runtime.python.main --target s100p --variant l --dry-run
```

Prepare and run on the actual X5 board:

```bash
bash samples/vision/yolo26_depth/model/download.sh --target x5 --variant n
bash samples/vision/yolo26_depth/runtime/python/run.sh --target x5 --variant n \
  --test-img samples/vision/yolo26_depth/test_data/bus.jpg --output /work/depth/x5-n
```

For S, select the exact target and variant; for example S600 lite l:

```bash
bash samples/vision/yolo26_depth/model/download.sh --target s600 --variant l
bash samples/vision/yolo26_depth/runtime/python/run.sh --target s600 --variant l \
  --output /work/depth/s600-l
```

Each output directory must be new. Replace `s600` with `s100` / `s100p` for the
corresponding asset. `--target auto` can use observed board identity, or infer
identity from an exact `--asset-id`; it does not guess a platform on a host.
The former positional shell variant (`run.sh l`) is replaced by explicit
`--variant l`; archived source scripts retain their historical interface.

<a id="expected-results"></a>
## Expected results and historical references

Python writes `log_depth.npy` (192×192 F32), `depth_native.npy` (original H×W F32),
`depth.png`, `overlay.png` and `report.json`; S lite additionally writes
`raw_logit.npy`. Native X5 also preserves `depth_native.f32`. Colors use the
2%/98% range, inverted TURBO, and 0.45 original / 0.55 depth overlay weights.
Reports record selection and actual local hashes; runtime version remains
`unknown` when the SDK does not expose it. A plausible visualization is not an
accuracy or metric-distance guarantee.

The source S root reported this mixed-profile single-image table. It is retained
as historical information, **not remeasured**:

| Variant | Profile | raw cosine vs FP32 | S100 latency ms | S100P latency ms | S600 latency ms |
|---|---|---:|---:|---:|---:|
| n | NV12 | 0.9996 | — | — | — |
| s | NV12 | 0.9984 | — | — | — |
| m | NV12 | 0.9996 | — | — | — |
| l | lite | 0.9997 | 11.0 | 8.1 | — |
| x | lite | 0.9997 | 20.6 | 13.7 | 10.8 |

Its all-pass ≥0.999 claim conflicts with s=0.9984. Its evaluator has a separate
latency table without complete per-row artifact binding. Do not merge those
numbers into one benchmark or infer acceptance. X5 HRT latency/FPS and the S
alternate table are preserved in the evaluator README. Source graph prose also
incorrectly placed exp/resize in graph; executable source performs them on CPU.
The [source audit](../../../docs/releases/unified-migration/2026-09-26-b8-yolo26-depth-source-review.md)
records these differences and the migration decisions.

<a id="directory"></a>
## Directory

```text
yolo26_depth/
├── model/                 # manifest-backed explicit download
├── runtime/python/        # stages, immutable context, binding, lazy runner, CLI, rendering
├── runtime/cpp/           # X5 stage API, SDK owner, tensor/IO helpers, launcher
├── conversion/            # export, calibration, 29 source YAMLs, X5/S compile
├── evaluator/             # three preparation protocols, offline metrics and comparisons
├── test_data/bus.jpg      # source image, preserved byte-for-byte
└── tests/                 # host fixtures and native pure/fake-SDK checks
```

<a id="entry-points"></a>
## Entry points and integration

- [Model](model/README.md): all assets, explicit preparation and provenance.
- [Python](runtime/python/README.md): CLI parameters, three-stage API, tensor contracts and errors.
- [C++](runtime/cpp/README.md): native dependencies, lifecycle, build/run and verification limits.
- [Conversion](conversion/README.md): checkpoint boundaries, calibration, resolved compiler inputs and known gaps.
- [Evaluator](evaluator/README.md): saved-array formats, protocols, metric definitions and historical tables.

API integration uses `RuntimeModelRunner` plus `Yolo26DepthTask`; `predict` is
exactly pre_process → forward → post_process, with no hidden second implementation.
For self-converted artifacts use explicit `--converted-model` together with
`--model-path` and the exact source `--asset-id` contract reference. The resulting
bytes are labeled user-converted; they do not inherit a publisher hash or measured
accuracy. S lite calibration constants must match the declared checkpoint.

[Archived X5](../../../platforms/x5/samples/vision/yolo26_depth/README.md) and
[archived S](../../../platforms/s/samples/vision/yolo26_depth/README.md) retain the
original implementations and records. They are provenance, not the unified entry.

<a id="license"></a>
## License

Code follows the repository [license](../../../LICENSE). Upstream Ultralytics
weights and SUN RGB-D data retain their own terms; neither is bundled here.
