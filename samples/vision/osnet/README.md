English | [简体中文](README_cn.md)

# OSNet person re-identification

<a id="overview"></a>
## Model and sources

OSNet uses omni-scale convolutional blocks and channel gates to encode an already-cropped person image into a 512-dimensional appearance feature. This sample uses official x1.0 Market1501 weights, three X5 quantization variants, and separately compiled/measured S100/S100P/S600 HBMs; spatial-rgb-late8 remains the X5 default. Sources: [official repository](https://github.com/KaiyangZhou/deep-person-reid), [ICCV paper](https://arxiv.org/abs/1905.00953), [official weights](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html).

<a id="directory"></a>
## Directory

```text
osnet/
├── model/          # Artifact preparation and identities
├── runtime/        # Python extraction and C++ verification
├── conversion/     # Export, calibration and compile configs
├── evaluator/      # Market1501 metrics and original evidence
├── test_data/      # Privately prepared fixed inputs
├── tests/          # Host behavior tests
└── third_party/    # Pinned upstream source and MIT license
```

<a id="support-matrix"></a>
## Support matrix

| Target | Variant | Language | Artifact |
| --- | --- | --- | --- |
| x5 | spatial-rgb-late8 | Python / C++ | Archive (research-only) |
| x5 | spatial-rgb-int16 | Python / C++ | available |
| x5 | spatial-rgb | Python / C++ | available |
| s100 | s100-f32-int16 | Python / C++ | Archive (research-only) |
| s100p | s100p-f32-int16 | Python / C++ | Archive (research-only) |
| s600 | s600-f32-int16 | Python / C++ | Archive (research-only) |
S100/S100P/S600 use separately validated nash-e/nash-m/nash-p HBMs. Artifacts are target-specific. Upstream x0.25/x0.5/x0.75 checkpoints exist (about 0.2/0.6/1.3 M parameters, 0.08/0.27/0.57 GFLOPs at 256×128). x1.0 is about 2.2 M / 0.98 GFLOPs. Other widths and IBN/AIN do not have RDK artifacts in this sample.


The new exact measurement is 2,169,508 inference parameters and 1.957750784 GFLOPs when each Conv/Linear MAC counts as two operations (0.978875392 GMAC). This differs from the upstream table’s one-MAC convention, not from the network width.

<a id="prerequisites"></a>
## Prerequisites

Use an X5 image supplying hbm_runtime compatible with libdnn 1.24.5 and HBRT 3.15.55.0, the recorded artifact environment. The original OS image version was not recorded. Python ≥3.10, NumPy and Pillow ≥9.1 are required; conversion also needs PyYAML. Preparation uses about 40 MB for the recovery TAR, BINs and fixed inputs. Use an existing environment; these scripts do not install dependencies.

All commands below run from the repository root.


S-series board testing used UCP/DNN 3.13.6 and HBRT 4.7.5, with board-supplied hbm_runtime, NumPy and Pillow (S100 Pillow 9.0.1 is supported; host export dependencies are in requirements-host.txt). S100/S100P used OS 4.0.5-Beta; S600 used 5.1.0.

<a id="quickstart"></a>
## Quick start

Run from the repository root. Download the Archive model with its pinned SHA-256 and prepare a person crop you have permission to use. Downloads contain models only. X5 example:

```bash
python3 samples/vision/osnet/model/download.py --target x5 --download
python3 samples/vision/osnet/runtime/python/main.py --target x5 \
  --variant spatial-rgb-late8 --test-img /absolute/path/person_crop.jpg --output feature.npy
python3 -c 'import numpy as np; f=np.load("feature.npy"); print(f.shape, f.dtype, np.linalg.norm(f))'
```

For S100/S100P/S600, prepare the corresponding `--target` and run with `--variant <target>-f32-int16`. Example:

```bash
python3 samples/vision/osnet/model/download.py --target s100 --download
python3 samples/vision/osnet/runtime/python/main.py --target s100 \
  --variant s100-f32-int16 --test-img /absolute/path/person_crop.jpg --output feature.npy
```

<a id="expected-results"></a>
## Expected results

For offline preparation, replace `--download` with `--candidate-dir /absolute/path/accepted-delivery`. See the [Archive upload receipt](evaluator/evidence/2026-11-02-archive-upload.json). The user approved local source integration and research-use catalog registration; remote push and website deployment remain pending.

feature.npy is an owned float32 array of shape (512,), with L2 norm about 1 by default. --raw retains unnormalized values. Detection and track association belong to the application.


Latest local acceptance (2026-10-30), default late8: full Market1501 Rank-1 94.0618%, mAP 83.3643%; FP32 ONNX 94.2399% / 83.5696%. Runtime concurrency 1/2: mean 8.189 / 15.048 ms, aggregate 121.936 / 132.422 FPS. C++ streams 1/2: mean E2E 8.768 / 15.071 ms, aggregate 114.040 / 132.530 FPS. Each condition has 3 rounds × 200 frames (C++ per stream), 20 warmups; native CPU workers, schedutil, BPU 996 MHz. The original BIN is reused. [New receipts](evaluator/evidence/2026-10-30-acceptance.json); [Workbench workflow](evaluator/README.md). New quantization compilation was not run; older variant results below remain historical.

| x1.0 variant | Rank-1 | mAP | C++ E2E P50 / P95 |
| --- | --- | --- | --- |
| FP32 reference | 94.2399% | 83.5696% | — |
| spatial-rgb-late8 | 94.0618% | 83.3643% | 8.614 / 9.641 ms |
| spatial-rgb-int16 | 94.2102% | 83.5551% | 9.018 / 10.040 ms |
| spatial-rgb | 93.3195% | 81.7215% | 1.670 / 3.488 ms |

Original 2026-10-29 measurements used 3,368 query + 15,913 gallery images, L2-normalized squared Euclidean distance and no reranking. C++ used 50 warmups, 3,000 iterations, batch 1, one stream, BPU 996 MHz and CPU schedutil. E2E covers decoded 64×128 RGB crop → Pillow-equivalent NEON resize to 128×256 → synchronous runtime → L2. Decoding, detection and matching are excluded; other crop sizes cannot inherit these timings.

Default mAP loss is 0.2053 percentage points; fast INT8 loses 1.8481 points. [Evidence and validation scope](evaluator/evidence/README.md).


### S-series local acceptance (2026-10-30)

| Target | Rank-1 | mAP | Runtime FPS 1 / 2 | C++ E2E ms 1 / 2 | C++ FPS 1 / 2 |
| --- | --- | --- | --- | --- | --- |
| S100 | 94.0914% | 83.5073% | 534.01 / 622.06 | 2.263 / 3.202 | 441.87 / 623.81 |
| S100P | 94.0914% | 83.5073% | 655.52 / 779.28 | 1.832 / 2.560 | 545.71 / 780.36 |
| S600 | 94.0914% | 83.5121% | 816.89 / 1008.27 | 1.515 / 1.973 | 660.05 / 1012.55 |

INT16 activation policy; the same full Market1501 dataset, FP32 baseline and one-percentage-point gate. Each board was measured separately on BPU core 0 (core_id=1). Runtime 1/2 means submission concurrency; C++ uses 1/2 native workers. Three rounds: Runtime 200 total frames per round, C++ 200 per stream per round, 20 warmup frames. Clock policies were unchanged. Accepted models were uploaded to Archive on 2026-11-02 for research use; the failed default INT8 baseline remains historical evidence and is not a recommended download.

[Full receipts](evaluator/evidence/2026-10-30-s-series-acceptance.json) · [S-series commands and acceptance](evaluator/S_SERIES.md).

<a id="entry-points"></a>
## Entry points

[Model preparation](model/README.md) · [Python](runtime/python/README.md) · [C++](runtime/cpp/README.md) · [Conversion](conversion/README.md) · [Evaluation](evaluator/README.md). matching.py contains OSNet.preprocess → infer → postprocess, composed by predict. main.py constructs the model directly. Help/list/dry-run do not load the SDK.

<a id="license"></a>
## License

Integration code follows the repository [Apache-2.0 license](../../../LICENSE). Upstream OSNet source is [MIT](third_party/LICENSE.deep-person-reid); retain its copyright and permission notices. The current Market1501 checkpoint is for research according to the [upstream author](https://github.com/KaiyangZhou/deep-person-reid/issues/520); neither it nor the converted models are cleared here for unrestricted commercial redistribution. The Market1501 package prohibits distribution and commercial use, so dataset images, decoded fixtures and the current retrieval cover are excluded from public source delivery. See the [license review and sources](third_party/LICENSE_REVIEW.md).
