[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive planning sample

<a id="overview"></a>
## Overview

DiffusionDrive combines a three-camera RGB panorama, LiDAR BEV histogram, ego status and explicit diffusion noise for trajectory planning. The source describes a two-step truncated diffusion decoder producing eight future ego poses, with auxiliary agent-state and seven-class BEV heads. This sample consumes prepared NAVSIM features and preserves the Python inference, visualization, five-case execution and float-reference comparison capabilities of the S branch. It does not prepare raw sensor data, compute a complete NAVSIM score or execute a vehicle-control command.

Source algorithm references: [official DiffusionDrive project](https://github.com/hustvl/DiffusionDrive), [CVPR2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html), and [NAVSIM](https://github.com/autonomousvision/navsim). The source does not pin the exact upstream checkpoint/export revision; these links are references, not artifact provenance.

<a id="support-matrix"></a>
## Support matrix

| Target | Published HBM | Runtime | Migration verification |
| --- | --- | --- | --- |
| S100P / nash-m | `s100p/diffusiondrive_r34_256x1024_s100p.hbm` | Python / hbm_runtime | Host fixtures passed; board not-run |
| S600 / nash-p | `s600/diffusiondrive_r34_256x1024_s600.hbm` | Python / hbm_runtime | Host fixtures passed; board not-run |
| S100 / X5 | None | Explicit rejection | No fallback |

There is no native C++ source for this sample. The two published HBM digests are preserved and verified on download/loading. `auto` requires recognized local identity or an explicit asset identity; unknown hosts no longer silently select S600. Host parity checks use supplied float arrays and synthetic runtime metadata, not real HBM execution.

<a id="prerequisites"></a>
## Prerequisites

For real inference prepare the matching S100P/S600 runtime with `hbm_runtime`, Python, NumPy and OpenCV, and explicitly download the target model. The [model guide](model/README.md) documents paths and checksums. Host inspection and offline comparison do not need the SDK. Neither runtime wrapper installs dependencies or downloads models.

The supplied NPZ inputs already contain camera/LiDAR/status/noise features. Do not substitute raw sensor images or point clouds; no raw NAVSIM feature builder is bundled. Conversion is optional for published artifacts and has missing export/calibration prerequisites described in [conversion](conversion/README.md).

<a id="quickstart"></a>
## Quickstart

From the repository root, inspect available models and a target contract without executing the SDK:

```bash
python3 -m samples.vision.diffusiondrive.runtime.python.main --list-models
python3 -m samples.vision.diffusiondrive.runtime.python.main --target s600 --dry-run
```

Explicitly prepare one model:

```bash
bash samples/vision/diffusiondrive/model/download.sh --target s600
```

On a prepared S600, run the default case with a new directory:

```bash
bash samples/vision/diffusiondrive/runtime/python/run.sh --target s600 --output outputs/diffusiondrive
```

Run all five source cases, or add `--dry-run` for host-only command/input inspection:

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s600 --output outputs/diffusiondrive_cases
```

For S100P, select `--target s100p` in both download and runtime commands. Its HBM is distinct; changing a filename does not retarget a model. See [runtime parameters and integration](runtime/python/README.md) before using external model paths.

<a id="expected-results"></a>
## Expected results

Each run writes physical quantized inputs, raw physical outputs, decoded `outputs.npz`, `result.png` and a provenance report. Decoded results contain trajectory `[1,8,3]`, thirty agent states/scores/masks and a seven-class BEV semantic prediction. All noise remains caller-supplied. Raw and decoded tensors have different schemas; retain both when diagnosing quantization.

The visualization combines the camera panorama, BEV semantics and LiDAR raster with orange trajectory, red filtered agents and blue ego vehicle. Gray denotes road, so a mostly gray BEV is not automatically a palette failure. Coordinate and class details are in [test data](test_data/README.md).

The following source S600 figure is historical, not a new migration measurement:

![Historical S600 DiffusionDrive display](test_data/reference_result.png)

| Historical case_017 | Historical case_042 |
| --- | --- |
| ![Intersection](test_data/case_017/result.png) | ![Dense traffic](test_data/case_042/result.png) |
| Historical case_073 | Historical case_099 |
| ![Boulevard](test_data/case_073/result.png) | ![Wide intersection](test_data/case_099/result.png) |

All six input/reference pairs and six result images are retained byte-for-byte. The [evaluator guide](evaluator/README.md#reference-results) preserves the complete original S100P/S600 accuracy/performance table, including one-thread latency, two-thread aggregate throughput and five-case S100P means. The [test-data guide](test_data/README.md) preserves the five-case S600 table. Accuracy comparisons used case_000, profiling case_017; neither was rerun here. Source CPU0ms/BPU-only claims remain historical.

<a id="directory"></a>
## Directory responsibilities

| Directory | Content |
| --- | --- |
| [model](model/README.md) | Exact target assets, SHA256SUMS, explicit downloads |
| [runtime/python](runtime/python/README.md) | CLI, batch entry, strict four-input/four-output binding, task stages and separate rendering |
| [conversion](conversion/README.md) | Two preserved OE3.7.0 PTQ configs and missing export/calibration prerequisites |
| [evaluator](evaluator/README.md) | Strict offline decoded-versus-float metrics, with shape and finite-value checks |
| [test_data](test_data/README.md) | Default and five deterministic NAVSIM feature/reference cases, historical images |
| [tests](tests) | Host source parity, quantization, CLI, batch and evaluator tests |

<a id="entry-points"></a>
## Entry points for people and agents

Use `DiffusionDriveTask.pre_process`, `forward`, `post_process`, or `predict`. The task handles planning tensor semantics only; SDK loading/scheduling, NPZ IO, download, rendering and metrics are outside it. The shared `NamedArrayRunner` preserves all named physical tensors and checks board/artifact identity. A [complete API example](runtime/python/README.md#integration-example) shows variables and input loading.

The refactor fixes source per-axis/scalar-zero-point handling, rejects malformed/negative quantization scales, prevents integer saturation overflow and rejects evaluator shape broadcasting. The evaluator's zero-norm cosine is explicitly undefined. Source output math and six-case rendering are compared on the host; these checks do not substitute for board validation. Original source remains in `platforms/s/samples/vision/diffusiondrive`.

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE). DiffusionDrive and NAVSIM assets remain subject to their own original terms. The included examples do not constitute a complete licensed NAVSIM dataset or a certified driving system.
