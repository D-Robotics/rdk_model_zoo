# Ultralytics YOLO — RDK X5 / S

[简体中文](README_cn.md)

<a id="overview"></a>
## Overview

This sample provides object detection, instance segmentation, pose estimation, classification and YOLO26 oriented boxes for RDK X5, S100, S100P and S600. YOLO detection heads predict classes and boxes at multiple scales; CPU decoding/filtering restores original-image coordinates. Segmentation, pose and OBB also expose masks, keypoints and angles. Model source project: [Ultralytics](https://github.com/ultralytics/ultralytics).

The maintained entry is `samples/vision/ultralytics_yolo`. Python binds inputs by target and selects task protocols by family, sharing preparation, rendering and evaluation. YOLO26 is one family within this sample. Standalone YOLOv5, YOLOE and yolo26_depth have distinct capabilities and are not presented as the same model here.

<a id="support-matrix"></a>
## Support and validation

States distinguish task and language. supported-verified refers only to the recorded fixed-input migration comparisons below, not all accuracy/performance or every current configuration.

| Scope | x5 | s100 | s100p | s600 |
|---|---|---|---|---|
| Python YOLOv8n / YOLO26n detect | supported-verified | supported-verified | supported-verified | supported-verified |
| Python other published scales/tasks | supported-not-run | supported-not-run | supported-not-run | supported-not-run |
| Python YOLOv9 seg c/e | supported-not-run | supported-not-run | supported-not-run | not-supported |
| Python YOLOv13 detect n/s/l/x | supported-not-run | not-supported | not-supported | not-supported |
| C++ detect/classify/pose/segment reference contracts | supported-not-run | supported-not-run | supported-not-run | supported-not-run |
| C++ OBB | not-supported | not-supported | not-supported | not-supported |

See the [model inventory](model/README.md) for published combinations: YOLOv5u/v10/12 detection; YOLOv8/11 detect/seg/pose/cls; YOLOv9 segmentation only c/e, with no t detection or segmentation on S600; YOLOv13 X5 only. YOLO26 has 25 assets per target (five tasks × n/s/m/l/x), consolidating existing assets rather than releasing 100 new models. C++ scope follows its [input/head contracts and limitations](runtime/cpp/README.md), not the full Python inventory.

Historical detection evidence: [P1](../../../docs/releases/unified-migration/2026-09-16-pilot-validation.md), [P2](../../../docs/releases/unified-migration/2026-09-16-p2-validation.md), covering YOLOv8n/YOLO26n on X5 8GB/4GB and the three S targets. These records do not extend to other tasks/scales, dataset accuracy, performance or local conversion. Historical measurements remain in the [X5 evaluator](../../../platforms/x5/samples/vision/ultralytics_yolo/evaluator/README.md) and [S evaluator](../../../platforms/s/samples/vision/ultralytics_yolo/evaluator/README.md); they are not remeasurements of the refactor.

| Runtime contract | X5 | S100 / S100P / S600 |
|---|---|---|
| Artifact | `.bin` / bayese | `.hbm` / nashe, nashm, nashp |
| NV12 input | One packed buffer | NHWC Y + UV |
| Python detection NMS default | 0.70 | 0.45; YOLOv10 is NMS-free |
| Classification CLI resize | YOLO26 stretch; others letterbox | Stretch |
| Classification filename tokens (not an input override) | YOLO26 224, others 640 | Public URLs use 224; S100/S100P v8/v11 retain 640 compatibility IDs |

<a id="prerequisites"></a>
## Prerequisites

Use a complete checkout, Python 3, NumPy, OpenCV, SciPy and PyYAML. Actual inference additionally needs `hbm_runtime` supplied by the matching board image. No dependencies are silently installed. Image/SDK versions must match your artifact and the validation records above; this sample does not establish one minimum image version for every target. Check storage and memory before choosing a large model; peak memory for every scale was not measured here, and successful download does not prove it will fit.

Runtime, download and host conversion environments are separate: C++ needs board development libraries, while ONNX/quantization compilation needs training and OE environments. See the subdirectories. Help/list/dry-run/download can run on a host without board inference. Hardware identity uses the [shared registry](../../../docs/release/platforms.json); recognized identity is not artifact availability or validation.

<a id="quickstart"></a>
## Quick start

Run from the repository root. Prepare the model with network access, then run on a matching X5 board; the input image is bundled in test_data.

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5.jpg
```
For other targets use matching preparation arguments and paths from [model instructions](model/README.md). Explicit `--model-path` never downloads; when omitted, the compatibility entry downloads missing default assets. Specify `--family` for custom filenames. `--target` aliases `--platform`; actual inference rejects unknown boards and target mismatches.

Without a board, inspect selections without downloading or inference:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --list-models
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls --dry-run
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --target x5 --task detect \
  --asset-id x5:ultralytics_yolo:yolov8n_detect_bayese_640x640_nv12.bin --dry-run
```
<a id="expected-results"></a>
## Expected results

The detection command prints model/input protocol and detections, writes the rendered image to `/tmp/yolov8n-x5.jpg`, and prints `[Saved]` on success. Boxes use original-image pixels and zero-based class IDs. Classification prints Top-K without an image; segmentation/pose/OBB fields are described in the [Python guide](runtime/python/README.md). A stale image is not a result of a failed binding. This retained historical detection illustration is not a new measurement:

![Historical detection illustration](test_data/ultralytics_YOLO_Detect_demo.jpg)

<a id="directory"></a>
## Directory responsibilities

```text
ultralytics_yolo/
├── model/          # published asset preparation
├── runtime/python/ # shared task APIs and CLI
├── runtime/cpp/    # detect/classify/pose/segment reference programs
├── conversion/    # export, calibration and X5/S compiler adapters
├── evaluator/     # dataset evaluation and batch CLI
├── test_data/     # input images, labels and historical illustrations
└── tests/         # host regression checks
```
<a id="entry-points"></a>
## Usage and source entry points

- [model](model/README.md) — Downloads, inventory, paths and digests.
- [runtime/python](runtime/python/README.md) — Task CLI, complete parameters, library integration and stage contracts.
- [runtime/cpp](runtime/cpp/README.md) — Task builds, positional arguments, lifecycle and measurement scope.
- [conversion](conversion/README.md) — Source weights, export, calibration, compilation and unverified prerequisites.
- [evaluator](evaluator/README.md) — COCO/ImageNet/DOTA, task evaluation and historical measurements.

Read `main.py` for arguments/files/rendering, `yolo_dispatch.py` for task selection, runner/binding for SDK/tensors, and task classes for preprocessing/inference/postprocessing. Detection DFL and YOLO26 direct LTRB are not interchangeable; see the [detection contract](DETECTION_CONTRACT.md). Input geometry must resolve from metadata or an explicit fallback, not filename guesses. YOLO26 OBB uses radians; X5 class-aware NMS/clipping differs from S. Corrected non-detection behavior still requires board accuracy revalidation.

Legacy `platforms/{x5,s}/samples/vision/ultralytics_yolo` and `ultralytics_yolo26` paths forward here for compatibility; historical tables, URLs and provenance remain. Do not infer that another pending sample is supported by this entry. New tasks require explicit output contracts, host regression coverage, bilingual documentation and scoped validation claims.

<a id="license"></a>
## License and provenance

Sample code follows the repository [Apache-2.0 LICENSE](../../../LICENSE), preserving file-level copyright notices. Check model weights and upstream training frameworks under their accompanying licenses separately; the repository code license does not automatically cover all weights. Artifact URLs and publisher digests come from manifests; an observed local hash cannot authenticate origin when no publisher hash is recorded.


Standalone S YOLO11 detection/pose/segmentation and S100 iMoonLab YOLOv13 source artifacts can now be prepared and selected by exact ID; numerical/C++ consolidation is not yet accepted. See [source asset routing and boundaries](model/README.md#standalone-assets).
