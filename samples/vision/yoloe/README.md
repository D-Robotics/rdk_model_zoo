# YOLOE PF instance segmentation

<a id="overview"></a>
## Overview

YOLOE provides prompt-free (PF) instance segmentation over a fixed, ordered 4585-class vocabulary. Version 11 uses DFL16 box regression and classwise NMS; version 26 uses direct LTRB and Top-K without NMS. These exports do not accept arbitrary text or visual prompts.

The canonical sample is `samples/vision/yoloe`.

Algorithm references and version context are preserved in the [X5 source overview](../../../platforms/x5/samples/vision/yoloe/README.md) and [S26 source overview](../../../platforms/s/samples/vision/yoloe26_seg/README.md).

<a id="support-matrix"></a>
## Support Matrix

| Variant | x5 | s100 | s100p | s600 | Canonical Python | Canonical C++ |
| --- | --- | --- | --- | --- | --- | --- |
| 11s | supported-not-run | not-supported* | not-supported | not-supported | X5; local float S route* | no, pending |
| 11m / 11l | supported-not-run | not-supported | not-supported | not-supported | X5 | no source X5 implementation |
| 26n/s/m/l/x | not-supported | not-supported* | not-supported* | not-supported | local float S route* | no, pending |

* For S, not-supported means the published artifact cannot run through this floating-output entry, not removal of the capability. S11 final HBM has mixed outputs and S26 public HBM declares quantized outputs. Explicit, hash-bound local float conversions can be selected, but that conversion route has not been built/verified. It is not supported-verified. S600 has no source support and never falls back to S100.

All current board checks are `not-run`. Canonical C++ and conversion integration are pending. Historical source board results do not verify this code.

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

X5 values are source single-thread libdnn Runtime records. S100 values are from 2026-09-08, 200 frames, warmup, thread_num=1/core_id=0, excluding preprocessing/postprocessing. S100P Runtime performance was not measured. See complete conditions and accuracy limits in [S26 evaluation](../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README.md).

<a id="directory"></a>
## Directory Layout

```text
yoloe/
├── model/             # explicit published-artifact preparation
├── runtime/python/    # CLI, binding, raw runner and three-stage task
├── test_data/         # source image and fixed vocabulary
├── tests/             # host fixtures, source comparisons and README execution
└── README.md
```

Canonical conversion/evaluator/C++ directories are still pending. The versioned source references below remain available; migration is not complete.

<a id="entry-points"></a>
## Entry Points

- [Model](model/README.md) — Publication identity, preparation and checksums.
- [Python](runtime/python/README.md) — Options, protocols, library API and troubleshooting.
- [Test data](test_data/README.md) — Image/vocabulary provenance.
- [X5 conversion](../../../platforms/x5/samples/vision/yoloe/conversion/README.md) / [S11 conversion](../../../platforms/s/samples/vision/yoloe11_seg/conversion/README.md) / [S26 conversion](../../../platforms/s/samples/vision/yoloe26_seg/conversion/README.md) — Original recipes; S recipes produce quantized outputs and still need the float-output route.
- [S11 C++](../../../platforms/s/samples/vision/yoloe11_seg/runtime/cpp/README.md) / [S26 C++](../../../platforms/s/samples/vision/yoloe26_seg/runtime/cpp/README.md) — Historical implementations; canonical port pending.
- [X5 evaluation](../../../platforms/x5/samples/vision/yoloe/evaluator/README.md) / [S26 evaluation](../../../platforms/s/samples/vision/yoloe26_seg/evaluator/README.md) — Historical records, not current acceptance.

<a id="license"></a>
## License

Sample code retains source Apache-2.0 notices and the repository [LICENSE](../../../LICENSE). Model weights and upstream YOLOE/Ultralytics software retain their respective licenses; the repository code license does not automatically license the weights. Version references are in the source overviews above.
