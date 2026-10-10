English | [简体中文](README_cn.md)

# Ultralytics YOLO — RDK X5 / S

<a id="overview"></a>

## Overview

This sample provides object detection, instance segmentation, pose estimation, classification and YOLO26 oriented boxes for RDK X5, S100, S100P and S600. YOLO detection heads predict classes and boxes at multiple scales; CPU decoding/filtering restores original-image coordinates. Segmentation, pose and OBB also expose masks, keypoints and angles. Model source project: [Ultralytics](https://github.com/ultralytics/ultralytics).

Python binds inputs by target and selects task protocols by family, sharing preparation, rendering and evaluation. YOLOv5, YOLOE and yolo26_depth use separate sample interfaces.

<a id="directory"></a>
## Directory structure

```text
ultralytics_yolo/
├── conversion/  # Export and quantization configuration
├── evaluator/  # Evaluation commands and metrics
├── model/  # Model files and download scripts
├── runtime/  # Python and native inference implementations
├── test_data/  # Example inputs
├── tests/  # Automated tests
├── DETECTION_CONTRACT.md  # Documentation
├── README.md  # English instructions
└── README_cn.md  # Chinese instructions
```

<a id="support-matrix"></a>
## Support and validation

States distinguish task and language. `supported` marks combinations with a published artifact and a runnable implementation; accuracy and performance figures are recorded separately in the evaluation guide.

| Scope | x5 | s100 | s100p | s600 |
|---|---|---|---|---|
| Python YOLOv8n / YOLO26n detect | supported | supported | supported | supported |
| Python other published scales/tasks | supported | supported | supported | supported |
| Python YOLOv9 seg c/e | supported | supported | supported | not-supported |
| Python YOLOv13 detect n/s/l/x | supported | not-supported | not-supported | not-supported |
| C++ detect/classify/pose/segment reference contracts | supported | supported | supported | supported |
| C++ OBB | not-supported | not-supported | not-supported | not-supported |

See the [model inventory](model/README.md) for published combinations: YOLOv5u/v10/12 detection; YOLOv8/11 detect/seg/pose/cls; YOLOv9 segmentation only c/e, with no t detection or segmentation on S600; YOLOv13 X5 only. YOLO26 has 25 assets per target (five tasks × n/s/m/l/x). C++ scope follows its [input/head contracts and limitations](runtime/cpp/README.md).

Select a published target/family/task/scale combination from the [model inventory](model/README.md). The [evaluator](evaluator/README.md) provides dataset scoring and reference measurements with their conditions.

| Runtime contract | X5 | S100 / S100P / S600 |
|---|---|---|
| Artifact | `.bin` / bayese | `.hbm` / nashe, nashm, nashp |
| NV12 input | One packed buffer | NHWC Y + UV |
| Python detection NMS default | 0.70 | 0.45; YOLOv10 is NMS-free |
| Classification CLI resize | YOLO26 stretch; others letterbox | Stretch |
| Classification filename tokens (not an input override) | YOLO26 224, others 640 | Public URLs use 224; S100/S100P v8/v11 retain 640 identifiers |

<a id="prerequisites"></a>
## Prerequisites

Use a complete checkout, Python 3, NumPy, OpenCV, SciPy and PyYAML. Actual inference additionally needs `hbm_runtime` supplied by the matching board image. Install dependencies explicitly and use the board image and SDK paired with the selected artifact. Choose a published target/family/task/scale combination from the [model inventory](model/README.md).

Runtime, download and host conversion environments are separate: C++ needs board development libraries, while ONNX/quantization compilation needs training and OE environments. See the subdirectories. Help/list/dry-run/download can run on a host. Hardware identity uses the [shared registry](../../../docs/release/platforms.json); select an artifact published for the matching target.

<a id="quickstart"></a>
## Quick start

Run from the repository root. Prepare the model with network access, then run on a matching X5 board; the input image is bundled in test_data.

```bash
bash samples/vision/ultralytics_yolo/model/download_model.sh \
  --platform x5 --family yolov8 --task detect --model-size n
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --model-path samples/vision/ultralytics_yolo/model/yolov8n_detect_bayese_640x640_nv12.bin \
  --label-file datasets/coco/coco_classes.names \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5.jpg
```
For other targets use matching preparation arguments and paths from [model instructions](model/README.md). Explicit `--model-path` never downloads; when omitted, the entry downloads missing default assets. Specify `--family` for custom filenames. `--target` aliases `--platform`; actual inference rejects unknown boards and target mismatches.

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

The detection command prints model/input protocol and detections, writes the rendered image to `/tmp/yolov8n-x5.jpg`, and prints `[Saved]` on success. Boxes use original-image pixels and zero-based class IDs. Classification prints Top-K without an image; segmentation/pose/OBB fields are described in the [Python guide](runtime/python/README.md). Example detection visualization:

![Reference detection illustration](test_data/ultralytics_YOLO_Detect_demo.jpg)

S-series detection visualization:

![Reference S delivery detection illustration](test_data/result_detect.jpg)

YOLO26 detection visualization with class IDs and scores:

![Reference S YOLO26 delivery detection illustration](test_data/result_detect_yolo26.jpg)

<a id="entry-points"></a>
## Usage and source entry points

- [model](model/README.md) — Downloads, inventory, paths and digests.
- [runtime/python](runtime/python/README.md) — Task CLI, complete parameters, library integration and stage contracts.
- [runtime/cpp](runtime/cpp/README.md) — Task builds, positional arguments, lifecycle and measurement scope.
- [conversion](conversion/README.md) — Source weights, export, calibration and target compilation.
- [evaluator](evaluator/README.md) — COCO/ImageNet/DOTA task evaluation and reference measurements.
- [test_data](test_data/README.md) — Bundled input images, display-label tables and example visualizations.

Read `main.py` for the visible construct-and-predict entry. ``cli.py`` owns option declarations, published-asset selection/listing, dry-run and result presentation; ``backend.py`` owns the runner, tensor binding and runtime metadata; the task files (`detect.py`/`segment.py`/`pose.py`/`obb.py`/`classify.py`) own preprocessing/inference/postprocessing and rendering inputs. Detection DFL and YOLO26 direct LTRB are separate protocols; see the [detection contract](DETECTION_CONTRACT.md). Input geometry resolves from metadata or an explicit fallback. YOLO26 OBB uses radians; X5 class-aware NMS/clipping differs from S.

<a id="license"></a>
## License and provenance

Sample code follows the repository [Apache-2.0 LICENSE](../../../LICENSE), preserving file-level copyright notices. Check model weights and upstream training frameworks under their accompanying licenses separately. Artifact URLs and publisher SHA-256 values are listed in the manifests.

The [model inventory](model/README.md) lists supported assets and runtime output requirements.

<a id="readable-example"></a>
## Readable example and custom models

This sample is one of the two readable model examples: the complete DFL
detection flow (initialization, `preprocess`, `infer`, `postprocess`,
`predict`) is visible in
[`runtime/python/detect.py`](runtime/python/detect.py); `main.py` stays a thin
entry that constructs the dispatched task model and calls `predict`. Each
protocol keeps its own task class (YOLO26 direct-LTRB, S-series NMS-free
YOLOv10, cls/seg/pose/obb). Custom-trained detectors connect through the
conversion flow plus `--model-path`/`--family` (and `--classes-num` with a
matching label file); predict accepts an image path or BGR array. An explicit
`--model-path` is treated as a custom model: without `--label-file` results
render class IDs (official COCO/ImageNet/DOTA sets are never applied
silently), and an explicit label file whose count differs from the bound
model's classes fails before inference. The image-path predict convenience
applies to the DFL detector (`detect.py`); cls/seg/pose/obb
keep their array interfaces. Three usage paths are described in
[docs/architecture/model-examples.md](../../../docs/architecture/model-examples.md).
