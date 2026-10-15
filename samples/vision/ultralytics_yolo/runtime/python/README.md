English | [简体中文](README_cn.md)

# Python runtime

<a id="overview"></a>
## Python inference

Run Ultralytics detection, segmentation, pose, classification and oriented-box models. `main.py` obtains the model class and configuration, constructs `Model(config)` and calls `predict`; each task module contains its preprocessing, inference and decoding.

<a id="directory"></a>
## Directory structure

```text
python/
├── backend.py  # Shared tensor binding, runner and runtime metadata
├── classify.py  # Classification task and Softmax/Top-K decode
├── cli.py  # Platforms, published assets, dispatch, listing and rendering
├── detect.py  # Detection tasks (DFL / YOLOv10 / YOLO26) with decode and shared geometry
├── main.py  # Command-line entry: construct the model and call predict
├── obb.py  # Rotated-box task and rotated NMS decode
├── pose.py  # Pose task and keypoint decode
├── run.sh  # Locate the Python entry and forward arguments
└── segment.py  # Segmentation tasks and mask decode
```

<a id="environment"></a>
## Board preparation

Copy or download the artifact for the same target as the board. X5 binds one
packed NV12 input; the S targets bind named NHWC Y and UV inputs. Before
inference the runtime reads the model metadata and checks batch, geometry,
dtype and output protocol; a filename suffix is never accepted as a
substitute for metadata.

The default model path comes from the platform Manifest. On a board, the
entry downloads a missing default model; an explicit
`--model-path` is never downloaded. For a host-side path check that does not
load `hbm_runtime`, use `--dry-run` or `--list-models` with an explicit
`--platform`. `--download` prepares the selected Manifest asset and exits.

The board image needs Python 3, NumPy, OpenCV, SciPy, and its matching
`hbm_runtime` module. Model files and the test image must be readable by the
calling user. Prepare the selected target’s model before inference.

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
`bash samples/vision/ultralytics_yolo/runtime/python/run.sh detect --platform s100`. It detects the board through
`/sys/class/boardinfo` when `--platform` is omitted; explicit target selection
is useful for listing, download, and dry-run on a host. Actual inference
rejects an unknown board or a target mismatch before loading the model.

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
| `--label-file` | str | `null` | Official defaults: COCO for detect/seg, a single `person` label for pose, ImageNet for cls, DOTA for obb. An explicit `--model-path` is a custom model: without this option the label-rendering paths show class IDs (pose drawing stays scores/keypoints — no class text); with it, the label count must equal the bound model's classes (json dict/list and line formats supported). |
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

Non-classification tasks default to letterbox. YOLO26 classification defaults to stretch on every target; other classification families use letterbox on X5 and stretch on S. Set `--reg`, `--strides`, `--mc`, geometry and class count to match the model output protocol. YOLO26 non-classification tasks reject DFL/keypoint/mask overrides differing from 16/17/32. YOLOv13 is published only on X5.

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

For dataset accuracy or performance measurement, use the [evaluator](../../evaluator/README.md). See [model preparation](../../model/README.md) for local paths and published combinations.

<a id="integration-example"></a>
## Library entry points

The DFL detection flow lives in `detect.py` (`YoloDetect`).
`predict` accepts a local image path or a BGR `uint8` array and never
modifies the array in place. Run this example from the repository root on
the matching S600 board, after replacing the model path with your local
YOLO11 detection artifact:

```python
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    YoloDetect, YoloDetectConfig)
from samples.vision.ultralytics_yolo.runtime.python.cli import (
    resolve_platform)

profile = resolve_platform("s600")
config = YoloDetectConfig(
    model_path="/models/yolo11n_nashp_640x640_nv12.hbm",
    platform=profile,
)
detector = YoloDetect(config)
result = detector.predict("samples/vision/ultralytics_yolo/test_data/bus.jpg")
boxes, scores, class_ids = result
print(boxes.shape, scores.shape, class_ids.shape)
```

The three stages can also be driven explicitly — `predict` chains exactly
these steps with the per-call transform:

```python
import cv2
import numpy as np

bgr_image = cv2.imread("samples/vision/ultralytics_yolo/test_data/bus.jpg")
if bgr_image is None:
    raise FileNotFoundError("Cannot read test image")
prepared = detector.preprocess(bgr_image)
raw = detector.infer(prepared)
result = detector.post_process(raw, transform=prepared.transform)
for staged, predicted in zip(result, detector.predict(bgr_image)):
    np.testing.assert_allclose(staged, predicted)
```

`YoloDetect` accepts an injected runner for alternate runtime
loaders. The runner is responsible for model execution; geometry preparation,
protocol binding, DFL decode, class-wise NMS, and coordinate restoration stay
in the shared task implementation. `pre_process` / `forward` / `post_process`
are aliases of the stage methods.
`YOLO26Detect` uses the shared image and runtime flow with direct LTRB decoding.

<a id="stage-io"></a>
## Code flow

`YoloDetect` and `YOLO26Detect` process inputs through `preprocess` → `infer` → `postprocess`. Retain the transform returned by each `preprocess` call and pass it to the corresponding `postprocess`; call one model instance serially. Detection, classification, segmentation, pose and OBB use these stage names, with `pre_process`, `forward` and `post_process` as aliases.

- `preprocess(img_or_path, image_format="BGR")` (alias `pre_process`) requires nonempty uint8 H×W×3 BGR, or a readable local image path. It returns `PreparedDetection.tensors` and a frozen `.transform` containing original/model/resized sizes, actual integer padding and per-axis scale. X5 tensors are packed NV12; S tensors are Y `(1,H,W,1)` and UV `(1,H/2,W/2,2)`, using metadata-derived H/W.
- `infer(prepared)` (alias `forward`) calls the runner once and returns role-keyed `RawOutputs`. The bound runner validates physical shape, dtype and finite values; it does not dequantize, activate, decode or change layout. Arrays keep the SDK dtype and borrow SDK buffers; the lifetime rules are in the shared stage rules below.
- `postprocess(raw, transform=prepared.transform)` (alias `post_process`) performs sigmoid/DFL or LTRB decoding, applicable NMS and coordinate restoration. Detection and DFL segmentation/pose require already-floating outputs; integer outputs or SCALE metadata fail at load time. No manual dequantization occurs in postprocessing.
- `predict(img_or_path)` composes those methods and returns owned result arrays. Plain semantic mappings from an injected runner must already hold floating values; use the bound raw carrier for physical floating tensors.

Prepared results support `[model_name]` mapping
access, and `infer`/`forward(prepared)` unwraps `.tensors`. Explicit
`post_process(outputs, original_width, original_height)` reconstructs the same
geometry. With both dimensions and a transform supplied,
they must agree. Per-image geometry is carried on the prepared object.
DFL segmentation shares this
transport; see its complete example below.
DFL pose, classification and YOLO26 OBB stages are documented below.

```text
main.py
  -> resolve_target / platform Manifest selection (cli.py for listing,
     dry-run, download preparation and presentation)
  -> cli.prepare_runtime_model -> Model(config) -> model.predict(image)
  -> ModelRunner + ModelBinding (input/output contract, shared SDK session)
  -> detect.resize_with_transform + NV12 input binding
  -> YoloDetect or YOLO26Detect decoder + NMS
  -> DetectionResult -> cli.present_result -> --img-save-path
```

`backend.py` identifies output roles by established shape/dtype contracts;
compiler enumeration names are opaque. `detect.py` records the actual
integer resize and padding so inverse boxes use the same transform.

<a id="stage-rules"></a>
### Rules shared by every task API

The task classes below (`YoloDetect`, `YOLO26Detect`, `YoloV10Detect`,
`YoloSeg`, `YOLO26Seg`, `YoloPose`, `YOLO26Pose`, `YOLO26OBB`, `YoloCls`)
differ only in model protocol and result shapes. The following rules apply to
all of them:

- **Floating model outputs only.** Every decoder requires the compiled model
  to emit unquantized FLOAT32 NHWC tensors. The loader rejects integer
  tensors, missing or incomplete shape/dtype metadata, SCALE quantization
  descriptors and non-finite values. Nothing dequantizes by
  hand; if binding fails, re-export the model with unquantized task outputs.
- **Roles bind by shape, never by output order.** Each tensor is
  identified by spatial shape and channel count. Ambiguous layouts are
  rejected; where an API offers an explicit role map, use it for custom class
  counts.
- **Protocols are not interchangeable.** The decoder follows the model's
  protocol — DFL distribution logits or direct LTRB distances — never the
  filename or output numbering.
- **Raw arrays borrow SDK buffers.** `infer` returns arrays that reference the
  runtime's own storage, which the next inference may overwrite. Finish
  postprocessing before the next SDK call, or copy any raw array you keep.
  `predict` and `postprocess` results own their storage and are safe to keep.
- **Pair every transform with its outputs.** Each call carries its own
  resize/padding on the returned `transform`; pass it to the matching
  `postprocess` call and drive one model instance serially. No "last image" is
  stored.
- **Numeric ranges.** Confidence thresholds must be finite in `(0,1)` and NMS
  thresholds finite in `[0,1]`; values are used exactly as supplied.
- **Empty results are valid.** Zero detections return arrays with the
  documented ranks and dtypes (`N=0`), and `masks=[]` where applicable.

Detection-style task APIs also accept an explicit
`(original_width, original_height)` pair in `post_process` instead of the
transform.

<a id="detection-io"></a>
### Detection input and output

`YoloDetect` (DFL) and `YOLO26Detect` (direct LTRB) share the input binding
and the result type; they differ in the box tensors and the decode.

The input is batch one, positive even H×W, `uint8` NV12. X5 binds one packed
buffer; S100/S100P/S600 bind named NHWC inputs `y=(1,H,W,1)` and
`uv=(1,H/2,W/2,2)`, with H/W taken from model metadata. Detection outputs
are already-floating NHWC tensors, six per model: at each configured stride s
(defaults 8, 16, 32) with grid (H/s, W/s), one classification tensor plus one
box tensor.

| Role | DFL families | YOLO26 |
|---|---|---|
| `cls_<s>` | `(1,H/s,W/s,C)` class logits, before sigmoid | same |
| `box_<s>` | `(1,H/s,W/s,4R)` DFL distribution logits, R=`--reg` bins, default 16 | `(1,H/s,W/s,4)` direct left/top/right/bottom distances in cell units |

DFL decoding applies one softmax per box side over the R bins and takes the
expected bin index; class scores use one sigmoid and the largest class per
cell. YOLO26 decoding applies no DFL softmax: the four channels are direct
distances from the cell center, the class is chosen in raw-logit space, and
sigmoid is applied only to the selected class. Both then apply the configured
NMS policy — `none`/`classwise`/`agnostic`, class-wise by default; S-series
YOLOv10 runs NMS-free — and map the boxes back to original-image coordinates.

The public result is `DetectionResult(boxes_xyxy, scores, class_ids)`:
`(N,4)` float32 original-image xyxy pixel boxes, `(N,)` float32 scores and
`(N,)` int64 zero-based class IDs. Empty detections keep these ranks/dtypes.

<a id="segmentation-api"></a>
## DFL segmentation library interface

For YOLOv8/9/11 segmentation, use `YoloSeg`; YOLO26 segmentation has a different
direct-box protocol and is not covered by this example. On a matching S100 board,
run from the repository root after preparing a compatible local segmentation
model through the [model instructions](../../model/README.md). Replace the example
absolute model path with your prepared artifact. The example uses the S platform default `nms_thres=0.45`; override it explicitly when needed.

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.segment import YoloSeg, YoloSegConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
segmenter = YoloSeg(YoloSegConfig(
    model_path="/models/yolo11n_seg_nashe_640x640_nv12.hbm",
    platform=resolve_platform("s100"),
    nms_thres=0.45,
))
prepared = segmenter.preprocess(image)
raw = segmenter.infer(prepared)
boxes, scores, ids, masks = segmenter.postprocess(raw, transform=prepared.transform)
expected = segmenter.predict(image)
for staged, predicted in zip((boxes, scores, ids), expected[:3]):
    np.testing.assert_allclose(staged, predicted)
assert len(masks) == len(expected[3])
for staged, predicted in zip(masks, expected[3]):
    np.testing.assert_array_equal(staged, predicted)
print(boxes.shape, scores.shape, ids.shape, [mask.shape for mask in masks])
```

The three methods have the same explicit `PreparedDetection`/`RawOutputs`
transport as detection. `YoloSeg(config, runner=...)` also supports an injected
runner. The factory reads actual input/output metadata and enforces target
identity before loading the real SDK. Inference never downloads a model.
`preprocess` (alias `pre_process`) accepts nonempty BGR uint8 H×W×3;
`postprocess` (alias `post_process`) takes either
`prepared.transform` or the explicit `(original_width, original_height)` pair.
Missing or conflicting geometry is an error.

The model must directly supply floating outputs: NHWC class logits
`(1,H/s,W/s,C)`, DFL box logits `(1,H/s,W/s,64)` and coefficients
`(1,H/s,W/s,32)` at strides 8/16/32, plus stride-4 prototypes
`(1,H/4,W/4,32)` or `(1,32,H/4,W/4)`. Published square
input geometry is required. Roles are bound from shapes or an explicit
`DFLSegmentationContract(output_roles=...)`; output enumeration order is
ignored. Postprocessing normalizes NCHW prototypes without manual
dequantization.

The result is `(boxes, scores, ids, masks)`: owned float32 `(N,4)` xyxy boxes
clipped to original-image `[0,width]`/`[0,height]`, float32 `(N,)` probabilities,
int64 `(N,)` class IDs, and N bool **ROI masks**, not full-image masks. Each mask
has shape `max(int(y2)-int(y1),0)` × `max(int(x2)-int(x1),0)`.
Empty results retain these array ranks/dtypes and return `masks=[]`. A degenerate
ROI preserves each zero-sized axis; no artificial one-pixel extent is added. Keep each mask paired with its own box.
The mask decode is the same as YOLO26's: combine
coefficients/prototypes, apply sigmoid, bilinearly resize the probabilities to
model size, crop by the model-space box, remove actual padding, resize to the
original image, threshold at `>0.5`, then take the box ROI. Thresholding after
interpolation keeps mask edges smooth; no morphological opening is applied.
The C++ sample thresholds at model resolution before cropping its ROI; this
decode removes the letterbox padding, resizes the probabilities to the
original image and thresholds there. The returned masks are per-box ROIs.
Geometry uses actual integer resize/padding, so the letterbox padding never
enters the mask.

<a id="pose-api"></a>
## DFL pose library interface

YOLOv8/11 pose shares the three-stage interface. Run from the repository root on
a matching S100 board after preparing the model locally as described under model/;
replace the absolute path below. Omitted NMS uses the platform default (S 0.45,
X5 0.70). YOLO26 pose uses direct boxes and is not covered by this DFL example.

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.pose import YoloPose, YoloPoseConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
pose = YoloPose(YoloPoseConfig(
    model_path="/models/yolo11n_pose_nashe_640x640_nv12.hbm",
    platform=resolve_platform("s100"),
))
prepared = pose.preprocess(image)
raw = pose.infer(prepared)
result = pose.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, pose.predict(image)):
    np.testing.assert_allclose(staged, predicted)
boxes, scores, class_ids, keypoints_xy, visibility = result
visible = visibility[..., 0] >= 0.5
print(boxes.shape, keypoints_xy.shape, visibility.shape, visible.sum())
```

`YoloPose(config, runner=...)` accepts an injected runner. Input handling and
raw-buffer lifetime follow the shared stage rules; explicit
`post_process(raw, original_width, original_height)` (alias of `postprocess`)
is also accepted, and missing/conflicting geometry fails. The nine outputs are NHWC class logits
`(1,H/s,W/s,1)`, DFL boxes `(1,H/s,W/s,64)` and keypoints `(1,H/s,W/s,51)`
(17 x/y/logit triplets) at strides 8/16/32. Square inputs, 16 DFL bins and
17 COCO points are required.

The five-tuple contains float32 `(N,4)` original-image boxes, float32 `(N,)`
detection probabilities, int64 `(N,)` class IDs (person=0), float32 `(N,17,2)`
original-image point coordinates and float32 `(N,17,1)` point probabilities.
NMS uses the same
indices for boxes and skeletons. Coordinates use actual resize/padding and clip
to `[0,width]`/`[0,height]`. Visibility receives one stable sigmoid; do not apply
another sigmoid or a zero-logit threshold to these probabilities. The example's
0.5 threshold is caller-side filtering; it does not alter coordinates or remove
points from the returned arrays.

The X5 adapter returns `(boxes, scores, keypoints)` with
`(N,17,3)` x/y/probability triplets. No separate S YOLO11Pose logits interface
is provided.

<a id="classification-api"></a>
## Classification stages

YOLOv8, YOLO11 and YOLO26 use the same classification pipeline. Run this example
from the repository root on the matching S600 board, replacing the absolute path
with the artifact prepared through the [model instructions](../../model/README.md).
Read the input geometry from SDK metadata. In particular, S100/S100P asset IDs containing `640`
link to URLs containing `224`; keep the published identity and inspect metadata.

```python
from pathlib import Path
import cv2
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.classify import YoloCls, YoloClsConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
classifier = YoloCls(YoloClsConfig(
    model_path="/models/yolo11n_cls_nashp_224x224_nv12.hbm",
    platform=resolve_platform("s600"),
    resize_type=0,
    topk=5,
))
inputs = classifier.preprocess(image)
raw = classifier.infer(inputs)
ranked = classifier.postprocess(raw)
assert ranked == classifier.predict(image)
print(ranked)
```

`preprocess` (alias `pre_process`) returns the nested NV12 input dictionary,
with no geometry carrier needed for classification. `infer` (alias `forward`)
calls the shared runner once and returns
`raw["logits"]` as the unchanged physical floating array, borrowing SDK storage.
Exactly one output is required: one 1000-class vector with optional
singleton axes (for example `(1,1000)` or `(1,1000,1,1)`); multidimensional
outputs require a leading batch size of one. Extra outputs, spatial
maps, batches, missing shape/dtype metadata, integer outputs and SCALE descriptors
are rejected.

`postprocess` (alias `post_process`) applies one SciPy Softmax and a descending
NumPy sort, then returns independent Python `(int class_id, float probability)`
pairs. Exact ties keep the NumPy sort order. Top-K must be a positive integer
(not a boolean); values above the class count return all classes, and zero,
negative and noninteger values raise an error. No labels, drawing or file writes
occur inside the stages. The CLI loads ImageNet labels and prints the results.

The library defaults to stretch (`resize_type=0`) for every family/target. The
CLI explicitly selects letterbox for X5 YOLOv8/11 classification, stretch for S,
and stretch for YOLO26 everywhere. Pass the same resize setting when comparing
library and CLI results.

<a id="v10-api"></a>
## S-series YOLOv10 without NMS

S-series YOLOv10 reuses the DFL detector's three stages with `nms="none"` fixed
in its binding contract. The CLI selects this `nms='none'` adapter for
S100/S100P/S600 v10; X5 v10 runs the DFL decoder with class-wise NMS. The two
dispatchers are selected by platform and are not interchangeable.

On a matching S600 board, run from the repository root after preparing the model
as described in [model preparation](../../model/README.md), then replace this
absolute path:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.detect import (
    YoloV10Detect, YoloV10DetectConfig,
)

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
detector = YoloV10Detect(YoloV10DetectConfig(
    model_path="/models/yolov10n_detect_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    score_thres=0.25,
))
prepared = detector.preprocess(image)
raw = detector.infer(prepared)
result = detector.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, detector.predict(image)):
    np.testing.assert_allclose(staged, predicted)
print(result.boxes.shape, result.scores.shape, result.class_ids.shape)
```

The six physical floating NHWC outputs bind by shape to class/16-bin DFL box
roles at strides 8/16/32; enumeration order does not choose their meaning. The
model input must be square. The result is owned
`(boxes, scores, class_ids)`, as in the common detection API.

All anchors meeting the confidence threshold remain, including overlapping
boxes. Results follow stride order and then grid traversal, with one best class
per anchor. There is no NMS, score sorting, Top-K limit or guarantee that boxes
cannot overlap. NMS thresholds have no effect here; a supplied binding contract
that enables NMS is rejected. `score_thres` uses the common detector's finite
`[0,1]` range. Its endpoint behavior is explicit: zero retains every finite-logit
anchor and one retains none.

Coordinate restoration uses the actual integer resized width/height and padding
rather than an ideal floating scale. Nonsquare images with rounding therefore
get coordinates corrected to the true geometry.

<a id="yolo26-pose-api"></a>
## YOLO26 pose stages

YOLO26 pose uses the same preprocessing, raw runner, explicit image context,
NMS and coordinate restoration as DFL pose. Its nine outputs instead contain
one class logit, **four direct LTRB distances**, and 17 `(x,y,visibility-logit)`
triplets per anchor at strides 8/16/32. Keypoint coordinates decode as
`(offset + grid_center) * stride`; DFL pose uses a different factor/offset rule.
The two decoders keep these formulas separate. Neither output-list order nor a
filename selects a protocol.

Run from the repository root on a matching S600 board, after preparing the model
through [model preparation](../../model/README.md) and replacing the absolute path:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.pose import YOLO26Pose, YOLO26PoseConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
pose = YOLO26Pose(YOLO26PoseConfig(
    model_path="/models/yolo26n_pose_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    nms_thres=0.45,
))
prepared = pose.preprocess(image)
raw = pose.infer(prepared)
result = pose.postprocess(raw, transform=prepared.transform)
for staged, predicted in zip(result, pose.predict(image)):
    np.testing.assert_allclose(staged, predicted)
boxes, scores, class_ids, keypoint_xy, visibility = result
print(boxes.shape, keypoint_xy.shape, visibility.shape)
```

Input must be nonempty BGR uint8 H×W×3; outputs must already be floating NHWC
with complete shape/dtype metadata. A DFL binding or ambiguous heads are
rejected.

The five results have the same shapes, dtypes and ownership as the DFL pose
API: float32
`(N,4)` boxes, float32 `(N,)` detection probabilities, int64 `(N,)` IDs,
float32 `(N,17,2)` points and float32 `(N,17,1)` point probabilities. The X5
adapter returns a list of `{box, score, kpts}` dictionaries with integer boxes;
the S
adapter returns a four-tuple with keypoints merged as `(N,17,3)`.

The YOLO26 library classes default to NMS 0.65; CLI defaults are X5 0.70 and
S 0.45. The
example explicitly matches S CLI behavior. Confidence thresholds are honored
exactly as supplied; with
actual integer geometry and the stable extreme-logit sigmoid, inputs with
rounding produce coordinates from the corrected geometry.

<a id="yolo26-segmentation-api"></a>
## YOLO26 segmentation stages

YOLO26 segmentation shares the DFL sample's stage transport, validated runner,
NMS and explicit geometry, but uses direct LTRB box distances. Its mask decode
is the DFL segmentation pipeline described above, ending in per-box ROI masks.

Run from the repository root on the matching S600 board after preparing a local
artifact through [model preparation](../../model/README.md); replace the path:

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.cli import resolve_platform
from samples.vision.ultralytics_yolo.runtime.python.segment import YOLO26Seg, YOLO26SegConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
segmenter = YOLO26Seg(YOLO26SegConfig(
    model_path="/models/yolo26n_seg_nashp_640x640_nv12.hbm",
    platform=resolve_platform("s600"),
    nms_thres=0.45,
))
prepared = segmenter.preprocess(image)
raw = segmenter.infer(prepared)
result = segmenter.postprocess(raw, transform=prepared.transform)
predicted = segmenter.predict(image)
for staged, repeated in zip(result[:3], predicted[:3]):
    np.testing.assert_allclose(staged, repeated)
assert len(result[3]) == len(predicted[3])
for staged, repeated in zip(result[3], predicted[3]):
    np.testing.assert_array_equal(staged, repeated)
boxes, scores, class_ids, masks = result
print(boxes.shape, [mask.shape for mask in masks])
```

The model requires a square input and ten outputs: class logits, four
LTRB distances and 32 mask coefficients at each of strides 8/16/32, plus 32
stride-4 prototypes. Heads are floating NHWC; prototypes may be floating NHWC or
NCHW when runtime shape proves that layout. In ambiguous custom-class layouts,
provide an explicit role map instead of relying on enumeration order.
Forward preserves the physical prototype layout; postprocess normalizes it.

The result owns float32 `(N,4)` boxes, float32 `(N,)` scores,
int64 `(N,)` IDs and a list of owned **boolean** ROI masks. For each clipped box,
use `x1,y1,x2,y2 = box.astype(int)` and place its mask at `[y1:y2,x1:x2]` in a
blank original-size image. A degenerate ROI keeps each zero-sized axis
(`(0,w)` or `(h,0)`); no detections return `(0,4)/(0,)/(0,)/[]`. The X5 adapter returns a boolean `(N,H,W)`
full-image mask stack; the S adapter returns ROI masks. `YOLO26Seg.predict` returns ROI masks for every target.

The decoder uses the supplied confidence threshold and selects class IDs from
logits before sigmoid; masks restore the original image geometry with the
recorded integer resize and padding.

<a id="yolo26-obb-stages"></a>
## YOLO26 rotated-box stages

Run this example from the repository root on S600 after preparing the OBB
artifact through [model preparation](../../model/README.md). Replace the local
model path. The bundled bus image only demonstrates calling the API; it is not
an aerial-object benchmark or an expected-detection assertion.

```python
from pathlib import Path
import cv2
import numpy as np
from samples.vision.ultralytics_yolo.runtime.python.obb import YOLO26OBB, YOLO26OBBConfig

image_path = Path("samples/vision/ultralytics_yolo/test_data/bus.jpg")
image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)
obb = YOLO26OBB(YOLO26OBBConfig(
    model_path="/models/yolo26n_obb_nashp_640x640_nv12.hbm",
    platform="s600",
))
prepared = obb.preprocess(image)
raw = obb.infer(prepared)
records = obb.postprocess(raw, transform=prepared.transform)
predicted = obb.predict(image)
assert len(records) == len(predicted)
for staged, repeated in zip(records, predicted):
    assert staged["id"] == repeated["id"]
    np.testing.assert_allclose(staged["rrect"], repeated["rrect"])
    np.testing.assert_allclose(staged["score"], repeated["score"])
for record in records:
    cx, cy, width, height, angle_radians = record["rrect"]
    print(record["id"], record["score"], record["rrect"])
```

The class reuses the detection class's shared image transport and raw runner
call (`preprocess`/`infer`, also reachable through the `pre_process`/`forward`
aliases) and adds its own rotated decode: `preprocess` returns input tensors
plus this image's actual integer resize and
padding. `infer` uses the common runner and returns borrowed raw SDK buffers;
finish postprocessing before the next inference or copy the buffers. It performs
no decoding or dequantization. `postprocess` (alias `post_process`) produces an
owned list of records:
`rrect=(cx,cy,width,height,angle_radians)`, floating `score`, integer `id`.
An empty detection set is `[]`.

The square model requires nine floating heads: 15 class logits, four direct
LTRB offsets and one **radian angle**, each at strides 8/16/32. Binding uses
shape/roles rather than output order. Absolute LTRB distances are used.
`angle_sign` multiplies the angle; `angle_offset` is in **degrees**, converted
before addition. `regularize=True` swaps width/height when width is smaller and
adds π/2. The exporter produces angles in radians.

| Behavior | X5 | S100 / S100P / S600 |
| --- | --- | --- |
| Rotated NMS | Per class, rotated IoU; suppress IoU ≥ threshold | OpenCV `NMSBoxesRotated`, all classes together |
| Angle after regularization | Wrapped to [−π/2, π/2) | No additional wrapping |
| Restored center and dimensions | Each clipped to original width/height | Not clipped |

Library defaults are confidence 0.25, NMS 0.2 and letterbox resize; the angle
controls must be finite. OpenCV
intersection errors propagate to the caller. Inverse mapping uses actual integer padding and per-axis scales,
correcting rounded-letterbox coordinates. Width and height are scaled axis-wise
with the angle unchanged: under unequal X/Y scales this is an
**approximate rotated rectangle**, not an exact transformed polygon.

Validate OBB geometry with the task-specific model classes and tensor
protocols shown above, keeping the same input dimensions, resize policy and
model-order DOTA labels used during inference.

<a id="troubleshooting"></a>
## Troubleshooting

* **`hbm_runtime` cannot be imported:** use the matching RDK board/system image
  and check its Python module path. Board inference runs on the board runtime
  image; the host OpenExplore compiler covers model compilation.
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
`--list-models`, and `--download` run on the host without board inference.

The YOLO inventory excludes duplicate standalone S variants; see [scope and output requirements](../../model/README.md#maintained-scope).
