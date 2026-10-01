# YOLOE PF instance segmentation

<a id="overview"></a>
## Overview

YOLOE provides prompt-free (PF) instance segmentation over a fixed, ordered 4585-class vocabulary. Version 11 (E11) uses DFL16 box regression — the 64 box channels decode into a 16-bin distribution per LTRB side at strides 8/16/32 — followed by classwise NMS. Version 26 (E26) uses no DFL (reg_max=1): its box head directly emits one LTRB distance per side, and candidate selection keeps global Top-K scores without NMS. These exports do not accept arbitrary text or visual prompts.

Both versions compose instance masks the same way: each kept candidate carries 32 mask coefficients that are linearly combined with the model's 32-channel 160×160 prototype features, and the combined map is thresholded at the sigmoid midpoint.

The canonical sample is `samples/vision/yoloe`.

Source attribution, inherited from the fixed X5/S sources:

- Paper: [YOLOE: Real-Time Seeing Anything](https://arxiv.org/pdf/2503.07465v1); official repo: [um-assn/yoloe](https://github.com/um-assn/yoloe) (X5 source)
- Base detector lineage: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (S11 source)

Algorithm references and version context are preserved in the X5 source overview (historical `../../../platforms/x5/samples/vision/yoloe/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) and S26 source overview (historical `../../../platforms/s/samples/vision/yoloe26_seg/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md).

<a id="support-matrix"></a>
## Support Matrix

| Variant | x5 | s100 | s100p | s600 | Canonical Python | Canonical C++ |
| --- | --- | --- | --- | --- | --- | --- |
| 11s | supported-not-run | not-supported* | not-supported | not-supported | X5; local float S route* | implemented; SDK/board not-run* |
| 11m / 11l | supported-not-run | not-supported | not-supported | not-supported | X5 | implemented X5 extension; SDK/board not-run |
| 26n/s/m/l/x | not-supported | not-supported* | not-supported* | not-supported | local float S route* | implemented; SDK/board not-run* |

* For S, not-supported means the published artifact cannot run through this floating-output entry, not removal of the capability. S11 final HBM has mixed outputs and S26 public HBM declares quantized outputs. Explicit, hash-bound local float conversions can be selected, but no compatible S float HBM has been compiled/verified. It is not supported-verified. S600 has no source support and never falls back to S100.

All current board checks are `not-run`. Canonical checkpoint export and calibration/configuration preparation are available; the C++ entry is implemented, while real SDK/OE acceptance remains pending. Historical source board results do not verify this code.

For native prerequisites, exact model selection, build/run commands and saved ROI results, see the [C++ workflow](runtime/cpp/README.md). The quick start below uses Python.

<a id="prerequisites"></a>
## Prerequisites

Python 3.10+, NumPy, OpenCV, SciPy and PyYAML. Real inference requires the matching board image’s `hbm_runtime`. Help, listing, dry-run and synthetic tensor tests work on a host. Host checks use Python 3.14.7; this is not board version acceptance.

The minimum X5 image version has not been established here. S26 source records RDK OS 4.0.5-Beta, UCP 3.13.6, HBRT 4.7.5 and OE 3.7.0; those are historical facts, not a new compatibility guarantee. Float32 class heads use about 154 MB per image, plus intermediates and masks. Peak memory and performance have not been measured.

<a id="quickstart"></a>
## Quick Start

```bash
# cwd: repository root
bash samples/vision/yoloe/model/download.sh --target x5 --variant 11s
python3 samples/vision/yoloe/runtime/python/main.py --target x5 --variant 11s
```

Run on X5. Preparation saves the original artifact under `model/x5/` and checks its publisher SHA-256. Inference uses the bundled `office_desk.jpg`, exits 0, prints JSON counts/scores/classes and writes `test_data/result.jpg`. Inference never downloads. `runtime/python/run.sh` forwards the same arguments.

Host inspection:

```bash
# cwd: repository root
python3 samples/vision/yoloe/runtime/python/main.py --list-models
python3 samples/vision/yoloe/runtime/python/main.py --target s100 --variant 26n --dry-run
```

A successful S dry-run proves selection only; it explicitly reports the separate float-conversion requirement.

<a id="expected-results"></a>
## Expected Results

Results contain original-image xyxy float32 boxes, sigmoid float32 probabilities and int64 PF class IDs. X5 11 retains full-image bool masks `[N,H,W]`; S11/26 return uint8 0/1 ROI lists, distinguished by `mask_layout`. Empty boxes have shape `[0,4]`, scores/IDs `[0]`.

No real model result is available this round; no fixed detection count is promised. Historical Runtime-only data is retained below; it is not unified Python end-to-end performance:

| Historical model | Target | Runtime latency / FPS |
| --- | --- | --- |
| YOLOE-11s PF | X5 | 146.16 ms / 6.84 |
| YOLOE-11m PF | X5 | 177.14 ms / 5.65 |
| YOLOE-11l PF | X5 | 189.97 ms / 5.26 |
| YOLOE-26n PF | S100 | 4.943 ms / 200.74 |
| YOLOE-26s PF | S100 | 9.944 ms / 100.08 |
| YOLOE-26m PF | S100 | 11.765 ms / 84.55 |
| YOLOE-26l PF | S100 | 13.417 ms / 74.18 |
| YOLOE-26x PF | S100 | 22.013 ms / 45.31 |

X5 values are source single-thread libdnn Runtime records. S100 values are from 2026-09-08, 200 frames, warmup, thread_num=1/core_id=0, excluding preprocessing/postprocessing. S100P Runtime performance was not measured. See complete conditions and accuracy limits in S26 evaluation (historical `../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md).

The two fixed S source publications each embedded one historical result illustration over the same bundled `office_desk.jpg`. They are restored here byte-exact under distinct names, so they can never be confused with the runtime-generated `test_data/result.jpg`:

![Historical S11 source result figure](test_data/source_s11_result_figure.jpg)

Historical result figure published with the fixed S11 source sample (platforms/s/samples/vision/yoloe11_seg (historical `../../../platforms/s/samples/vision/yoloe11_seg/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md), rdk_s `380e1a2b`). That source delivery runs only on S100 and its only published runnable artifact is a quantized 11s PF HBM; the source does not record which run produced the figure.

![Historical S26 source result figure](test_data/source_s26_result_figure.jpg)

Historical example published with the fixed S26 source (platforms/s/samples/vision/yoloe26_seg (historical `../../../platforms/s/samples/vision/yoloe26_seg/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)): recorded outputs from the released quantized S100 YOLOE-26n PF model, labels exported in PF checkpoint class-ID order, as stated by the source caption.

Both are historical quantized S publication results — not output of this sample's floating route, not expected results for the current code, and not accuracy/AP evidence.

<a id="directory"></a>
## Directory Layout

```text
yoloe/
├── model/             # explicit published-artifact preparation
├── conversion/        # ONNX checks, calibration, target YAML and optional compile
├── evaluator/         # explicit category mapping, COCO metrics and prediction export
├── runtime/python/    # CLI, binding, raw runner and three-stage task
├── runtime/cpp/       # reusable three-stage C++ library, SDK adapter, CLI, E11/E26 decoding
├── test_data/         # source image, fixed vocabulary, historical source figures
├── tests/             # host fixtures, source comparisons and README execution
└── README.md
```

Canonical checkpoint export, conversion preparation and evaluation workflows are available below; the canonical C++ entry is implemented, with real SDK compilation unverified. Float export checks do not establish real compiler or board acceptance.

<a id="entry-points"></a>
## Entry Points

- [Model](model/README.md) — Publication identity, preparation and checksums.
- [Conversion](conversion/README.md) — Checkpoint export, float-output checks, target-specific calibration, compiler commands and artifact records.
- [Evaluation](evaluator/README.md) — Explicit PF mapping, CPU/board backends, COCO metrics, provenance and historical benchmarks.
- [Python](runtime/python/README.md) — Options, protocols, library API and troubleshooting.
- [Test data](test_data/README.md) — Image/vocabulary provenance and historical source figures.
- X5 conversion (historical `../../../platforms/x5/samples/vision/yoloe/conversion/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) / S11 conversion (historical `../../../platforms/s/samples/vision/yoloe11_seg/conversion/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) / S26 conversion (historical `../../../platforms/s/samples/vision/yoloe26_seg/conversion/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) — Original recipes; S recipes produce quantized outputs. The canonical preparation entry retains float-output nodes, with actual compiled precision still unverified.
- [Canonical C++ progress](runtime/cpp/README.md) — Reusable three-stage C++ library with owned NV12 inputs, float binding and E11/E26 decoding/masks; the SDK adapter is implemented; native board/model/vocabulary preflight is implemented; publication selection, CLI and result records are implemented; real SDK compilation and board checks remain not-run.
- S11 C++ (historical `../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) / S26 C++ (historical `../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) — Historical implementations with preserved source capabilities and measurements.
- X5 evaluation (historical `../../../platforms/x5/samples/vision/yoloe/evaluator/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) / S26 evaluation (historical `../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md) — Historical records, not current acceptance.

<a id="license"></a>
## License

Sample code retains source Apache-2.0 notices and the repository [LICENSE](../../../LICENSE). Model weights and upstream YOLOE/Ultralytics software retain their respective licenses; the repository code license does not automatically license the weights. Version references are in the source overviews above.
