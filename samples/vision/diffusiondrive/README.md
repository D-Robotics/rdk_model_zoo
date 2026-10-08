[English](README.md) | [简体中文](README_cn.md)

# DiffusionDrive planning sample

<a id="overview"></a>
## Overview

DiffusionDrive plans future vehicle trajectories from camera, LiDAR and ego-state features. Its truncated diffusion decoder predicts eight future poses, with auxiliary agent-state and BEV semantic heads. This sample runs prepared NAVSIM features on RDK S100P and S600.

References: [official DiffusionDrive project](https://github.com/hustvl/DiffusionDrive), [CVPR2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Liao_DiffusionDrive_Truncated_Diffusion_Model_for_End-to-End_Autonomous_Driving_CVPR_2025_paper.html), [NAVSIM](https://github.com/autonomousvision/navsim).

<a id="directory"></a>
## Directory structure

```text
diffusiondrive/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Support matrix

| Target | Published HBM | Runtime | Preparation |
| --- | --- | --- | --- |
| S100P / nash-m | `s100p/diffusiondrive_r34_256x1024_s100p.hbm` | Python / hbm_runtime | supported |
| S600 / nash-p | `s600/diffusiondrive_r34_256x1024_s600.hbm` | Python / hbm_runtime | supported |
| S100 / X5 | None | Explicit rejection | No fallback |

Runtime language: Python. Use the published S100P or S600 HBM and the matching board SDK. Downloading and loading check the published HBM digest; select the target explicitly when preparing models on a host.

<a id="prerequisites"></a>
## Prerequisites

For real inference prepare the matching S100P/S600 runtime with `hbm_runtime`, Python, NumPy and OpenCV, and explicitly download the target model. The [model guide](model/README.md) documents paths and checksums. Host inspection and offline comparison do not need the SDK. Prepare dependencies and model files with the commands below.

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

Run all five source cases, or add `--dry-run` for command/input inspection:

```bash
bash samples/vision/diffusiondrive/runtime/python/run_all_cases.sh --target s600 --output outputs/diffusiondrive_cases
```

For S100P, select `--target s100p` in both download and runtime commands. Its HBM is distinct; changing a filename does not retarget a model. See [runtime parameters and integration](runtime/python/README.md) before using external model paths.

<a id="expected-results"></a>
## Expected results

Each run writes physical quantized inputs, raw physical outputs, decoded `outputs.npz`, `result.png` and a provenance report. Decoded results contain trajectory `[1,8,3]`, thirty agent states/scores/masks and a seven-class BEV semantic prediction. All noise remains caller-supplied. Raw and decoded tensors have different schemas; retain both when diagnosing quantization.

The visualization combines the camera panorama, BEV semantics and LiDAR raster with orange trajectory, red filtered agents and blue ego vehicle. Gray denotes road, so a mostly gray BEV is not automatically a palette failure. Coordinate and class details are in [test data](test_data/README.md).

S600 reference visualization:

![Reference S600 DiffusionDrive display](test_data/reference_result.png)

| Reference case_017 | Reference case_042 |
| --- | --- |
| ![Intersection](test_data/case_017/result.png) | ![Dense traffic](test_data/case_042/result.png) |
| Reference case_073 | Reference case_099 |
| ![Boulevard](test_data/case_073/result.png) | ![Wide intersection](test_data/case_099/result.png) |

The six input/reference pairs include S600 visualizations. The [evaluator guide](evaluator/README.md#reference-results) reports S100P/S600 accuracy and performance, including one-thread latency, two-thread aggregate throughput and five-case S100P means. The [test-data guide](test_data/README.md) gives five-case S600 results. Accuracy comparisons use case_000; profiling uses case_017. The model runs every segment on BPU with CPU inference time 0.0 ms.

Inputs are a three-camera RGB panorama, a LiDAR BEV histogram, ego status and explicit diffusion noise. Prepare these NAVSIM features before inference. The outputs support trajectory visualization and float-reference comparison; a complete NAVSIM score requires its dataset evaluator.

<a id="entry-points"></a>
## Entry points for people and agents

Use `DiffusionDrivePlanner.predict`, or the stages `preprocess` → `infer` → `postprocess` that it composes; the `pre_process`/`forward`/`post_process` spellings remain importable aliases of the same implementation. The task handles planning tensor semantics only; SDK loading/scheduling, NPZ IO, download, rendering and metrics are outside it. The shared `NamedArrayRunner` preserves all named physical tensors and checks board/artifact identity. A [complete API example](runtime/python/README.md#integration-example) shows variables and input loading.

Quantization checks per-axis scales and scalar zero points, rejects malformed or negative scales, and clips integer values before casting. The evaluator requires matching shapes; cosine similarity is undefined when either vector has zero norm.

<a id="license"></a>
## License

Sample code follows the repository [Apache-2.0 license](../../../LICENSE). DiffusionDrive and NAVSIM assets remain subject to their own original terms. Obtain complete datasets using the NAVSIM data preparation procedure and license.
