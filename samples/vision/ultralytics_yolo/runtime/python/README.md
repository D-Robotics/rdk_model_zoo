# Python runtime

[简体中文](README_cn.md)

This is the board-side entry point for the shared Ultralytics YOLO sample. It
loads a compiled `.bin` (X5) or `.hbm` (S100/S100P/S600) model through the
`hbm_runtime` supplied by the RDK system image, prepares one BGR image as the
selected target's NV12 input, runs the task decoder, and saves a rendered
result image for detect/seg/pose/obb. Classification prints Top-K instead. The script does not install Python packages. Export and compiler
steps belong in [`conversion/README.md`](../../conversion/README.md).

<a id="environment"></a>
## Board preparation

Copy or download the artifact for the same target as the board. X5 binds one
packed NV12 input; the S targets bind named NHWC Y and UV inputs. The runtime
reads model metadata and checks batch, geometry, dtype, and output protocol
before inference. A filename suffix is not accepted as a substitute for model
metadata.

The default model path comes from the platform Manifest. On a board, the
historical entry downloads a missing default model; an explicit
`--model-path` is never downloaded. For a host-side path check that does not
load `hbm_runtime`, use `--dry-run` or `--list-models` with an explicit
`--platform`. `--download` prepares the selected Manifest asset and exits.

The board image needs Python 3, NumPy, OpenCV, SciPy, and its matching
`hbm_runtime` module. Model files and the test image must be readable by the
calling user. The runtime never silently falls back to another platform or
model when a requested artifact is missing.

<a id="usage"></a>
## Shortest detection run

Run from the repository root on the matching board:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img samples/vision/ultralytics_yolo/test_data/bus.jpg \
  --img-save-path /tmp/yolov8n-x5-detect.jpg
```

For S100/S100P/S600 use the corresponding platform. To use a compiled file
you prepared yourself, pass it explicitly:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

The convenience script accepts the same options after a task name:
`bash samples/vision/ultralytics_yolo/runtime/python/run.sh detect --platform s100`. It detects the board when
`--platform` is omitted; explicit target selection is useful for listing,
download, and dry-run on a host. Actual inference rejects an unknown board or
a target mismatch before loading the model.

The following commands cover the other tasks and the default entry. With no arguments, the board is detected and yolo11 default-scale detection uses bus.jpg. Supply your own aerial image for OBB. Run each line on its matching board; this is not a script to execute all targets on one board.

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolo11 --task seg --img-save-path /tmp/yolo-seg.jpg
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s100 --family yolov8 --task pose --img-save-path /tmp/yolo-pose.jpg
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls \
  --test-img samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg --topk 5
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolo26 --task obb \
  --test-img /data/aerial.jpg --img-save-path /tmp/yolo-obb.jpg
```

## Task and model selection

`--family` and `--model-size` select a published Manifest combination; omit
the size to use that family/platform's default. `--asset-id` accepts the exact
`group:sample:filename` reference printed by `--list-models`. `--model-path`
selects a local artifact and does not download it; custom filenames should be
paired with `--family` so the decoder is known.

The common task names are `detect`, `seg`, `pose`, `cls`, and `obb` where the
selected family publishes them. YOLOv8/YOLO11 detection uses three feature
levels of DFL logits (`reg=16`). YOLO26 detection uses direct LTRB outputs at
strides 8/16/32 and therefore has a separate binding/decoder. Do not pass a
YOLO26 artifact to a DFL decoder or infer its protocol from output numbering.
The current representative board evidence covers YOLOv8n and YOLO26n
detection; it does not certify every model scale or task.

Registered families and their decoder protocol:

| family | tasks | decoder |
|---|---|---|
| `yolo26` | detect, seg, pose, cls, obb | direct-LTRB |
| `yolov5u` | detect | DFL |
| `yolov8` | detect, seg, pose, cls | DFL |
| `yolov9` | detect, seg | DFL |
| `yolov10` | detect | DFL; NMS-free decode on S-series |
| `yolo11` | detect, seg, pose, cls | DFL (default family) |
| `yolo12` | detect | DFL |
| `yolov13` | detect | DFL |

<a id="parameters"></a>
## CLI parameters

The Default column shows parser values. `null` means resolved later from platform, task or artifact; it does not disable the option.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--platform` / `--target` | str | `null` | Auto-detect board; auto/x5/s100/s100p/s600. Explicit preparation can select another target; inference cannot. |
| `--task` | str | `detect` | detect/seg/pose/cls/obb; availability depends on family. |
| `--family` | str | `null` | Infer recognized local filename, otherwise yolo11. Conflicting explicit family is rejected. |
| `--model-size` | str | `null` | Family/task default from the published inventory; see model README. |
| `--model-path` | str | `null` | Explicit compiled file; never automatically downloaded. |
| `--asset-id` | str | `null` | Exact group:sample:filename reference; constrains manifest selection. |
| `--input-shape` | HxW | `null` | Fallback input H×W only when runtime metadata is absent. |
| `--test-img` | str | `samples/vision/ultralytics_yolo/test_data/bus.jpg` | Readable BGR image loaded by OpenCV. |
| `--label-file` | str | `null` | COCO for detect/seg/pose, ImageNet for cls, DOTA for obb; override for custom class order. |
| `--img-save-path` | str | `result.jpg` | Rendered detect/seg/pose/obb image, relative to caller cwd; cls does not write it. |
| `--score-thres` | float | `0.25` | Detection confidence filter; not used by classification. |
| `--nms-thres` | float | `null` | Resolve by target/task; X5 detection 0.70, S detection 0.45. S YOLOv10 detection is NMS-free. |
| `--resize-type` | int | `null` | 0 stretch, 1 letterbox; resolved defaults described below. |
| `--classes-num` | int | `null` | Use task config default; override detect/seg/obb only to match a custom graph. |
| `--strides` | comma-separated ints | `[8, 16, 32]` | Feature-map strides, e.g. 8,16,32. |
| `--mc` | int | `32` | Segmentation mask coefficients; YOLO26 requires 32. |
| `--angle-sign` | float | `1.0` | OBB angle multiplier. |
| `--angle-offset` | float | `0.0` | OBB angle offset in degrees. |
| `--regularize` | int | `1` | OBB rotated-rectangle regularization, 0 or 1. |
| `--reg` | int | `16` | DFL regression bins; changing this does not convert a direct-LTRB graph. |
| `--nkpt` | int | `17` | Pose point count for DFL families; YOLO26 requires 17. |
| `--topk` | int | `5` | Number of classification results. |
| `--kpt-conf-thres` | float | `0.5` | Pose drawing visibility threshold; not a tensor-binding parameter. |
| `--priority` | int | `0` | BPU scheduling priority, 0–255. |
| `--bpu-cores` | space-separated ints | `[0]` | One or more core indexes, e.g. --bpu-cores 0 1. |
| `--list-models` | flag | `false` | Print supported models and exact references, then exit. |
| `--dry-run` | flag | `false` | Resolve selection only; no download or inference. |
| `--download` | flag | `false` | Prepare selected published model and exit without inference. |

Non-classification tasks default to letterbox. YOLO26 classification defaults to stretch on every target; other classification families use letterbox on X5 and stretch on S. Geometry and class counts must match the model; `--reg`, `--strides` and `--mc` cannot force compatibility. YOLO26 non-classification tasks reject DFL/keypoint/mask overrides differing from 16/17/32. YOLOv13 is published only on X5.

<a id="results"></a>
## Results

Exit status 0 means the command completed; an empty detection list can be valid. Detect/seg/pose/obb write `--img-save-path`, create its parent directory and print `[Saved]`; an existing result at that path is overwritten. Classification prints results without saving an image. Stage APIs do not draw or save files.

| Task | `predict` return | Coordinates / meaning |
|---|---|---|
| detect | `DetectionResult(boxes_xyxy, scores, class_ids)`, tuple-compatible | `(N,4)` original-image pixel boxes, `(N,)` scores and zero-based class IDs |
| seg | `(boxes, scores, ids, masks)` | Original-image pixel boxes and matching instance masks; preserve instance order |
| pose | `(boxes, scores, ids, xy, confidence)` | Original-image boxes and point coordinates; published pose models have 17 points, with separate point confidence |
| cls | List of `(class_id, probability)` | Score-sorted Top-K after Softmax, not raw logits |
| obb | List of dictionaries: `rrect`, `score`, `id` | `rrect=(cx,cy,w,h,angle)`; original-image center/size, angle in radians |

One rendered image is not dataset accuracy or performance validation; use the [evaluator](../../evaluator/README.md). See [model preparation](../../model/README.md) for local paths and published combinations.

<a id="integration-example"></a>
## Library entry points

Run this example from the repository root on the matching S600 board, after replacing the model path with your local YOLO11 detection artifact:

```python
import sys
from pathlib import Path
import cv2

runtime_dir = Path("samples/vision/ultralytics_yolo/runtime/python").resolve()
sys.path.insert(0, str(runtime_dir))
from yolo_platform import resolve_platform
from yolo_detect import YoloDetect, YoloDetectConfig

profile = resolve_platform("s600")
config = YoloDetectConfig(
    model_path="/models/yolo11n_nashp_640x640_nv12.hbm",
    platform=profile,
)
bgr_image = cv2.imread("samples/vision/ultralytics_yolo/test_data/bus.jpg")
if bgr_image is None:
    raise FileNotFoundError("Cannot read test image")
detector = YoloDetect(config)
boxes, scores, class_ids = detector.predict(bgr_image)
print(boxes.shape, scores.shape, class_ids.shape)
```

`YoloDetect` accepts an injected runner for host tests and alternate runtime
loaders. The runner is responsible for model execution; geometry preparation,
protocol binding, DFL decode, class-wise NMS, and coordinate restoration remain
in the shared task implementation. `YOLO26Detect` uses the shared image and
runner orchestration with its reviewed direct-LTRB decoder. Legacy X5 and S
modules keep their historical class names and tuple shapes while forwarding to
these maintained paths.

<a id="stage-io"></a>
## Code flow

`predict` composes preprocessing, one inference call and postprocessing. File reads, logs, labels and rendering belong to the CLI/helpers. Manual stage calls must preserve the same image geometry and transform; do not interleave images through one stateful model instance.

- `pre_process(img, image_format="BGR")`: H×W×3 image to uint8 NV12 tensors grouped by model/input name. X5 uses a packed buffer; S binds Y `(1,H,W,1)` and UV `(1,H/2,W/2,2)`. H/W come from model binding.
- `forward(inputs)`: run the model and return raw output mappings; no drawing, saving or detection filtering. Postprocessing interprets outputs through binding rather than treating physical output indexes as semantic roles.
- Detection `post_process(outputs, ori_img_w, ori_img_h, ..., transform=...)`: DFL or LTRB decode, applicable NMS and coordinate restoration, returning the result above. `pre_process_with_transform` also returns the actual resize/padding transform; `pre_process` keeps the historical tensor-only return.
- Seg/pose also decode masks/keypoints, cls applies Softmax and Top-K, and obb decodes rotated boxes. Their postprocessing signatures differ; prefer each task's `predict` or consult that module's docstrings for manual stage use.


```text
main.py
  -> resolve_target / platform Manifest selection
  -> yolo_dispatch.get_task_types / create_runtime_model
  -> ModelRunner + ModelBinding (input/output contract)
  -> geometry.resize_with_transform + NV12 input binding
  -> YoloDetect or YOLO26Detect decoder + NMS
  -> DetectionResult -> visualize -> --img-save-path
```

`model_binding.py` identifies output roles by reviewed shape/dtype contracts;
compiler enumeration names are opaque. `geometry.py` records the actual
integer resize and padding so inverse boxes use the same transform. The
finite protocols and old-to-new symbol map are in
[`DETECTION_CONTRACT.md`](../../DETECTION_CONTRACT.md).

<a id="troubleshooting"></a>
## Troubleshooting

* **`hbm_runtime` cannot be imported:** use the matching RDK board/system image
  and check its Python module path. Installing a host OpenExplore compiler does
  not provide board runtime support.
* **Unknown board or target mismatch:** omit `--platform` on the board or pass
  the board's real target. A CLI flag cannot override an unknown hardware
  identity.
* **Model not found:** use `--list-models` to see the exact Manifest reference,
  `--download` to prepare a published asset, or provide an existing
  `--model-path`. Explicit paths are never replaced by a similar download.
* **Input/output contract rejected:** verify target, artifact family, static
  even input dimensions, NV12 roles, and the DFL/LTRB output protocol. Check
  `--reg`, `--strides`, and `--classes-num` against the compiled graph.
* **No detections or unexpected boxes:** check the test image color/order,
  `--resize-type`, `--score-thres`, and `--nms-thres`. The decoder restores
  coordinates to the original image; it does not treat output names as a
  geometry declaration.
* **Cannot save the result:** ensure the parent directory of
  `--img-save-path` is writable and that the path is relative to the caller
  when it is not absolute.

Use `python samples/vision/ultralytics_yolo/runtime/python/main.py --help` for the complete CLI. `--help`, `--dry-run`,
`--list-models`, and `--download` are host-safe paths that do not run board
inference.


Standalone S YOLO11 detection/pose/segmentation and S100 iMoonLab YOLOv13 source artifacts can now be prepared and selected by exact ID; numerical/C++ consolidation is not yet accepted. See [source asset routing and boundaries](../../model/README.md#standalone-assets).
