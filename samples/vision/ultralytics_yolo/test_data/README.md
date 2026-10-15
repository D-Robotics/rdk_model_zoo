English | [简体中文](README_cn.md)

# Ultralytics YOLO test data

This directory provides eleven fixed files: input images, runtime display labels
and reference result illustrations.

1. **Input images** — `bus.jpg` and `zebra_cls.jpg`, used by the Python runtime, C++ and conversion `--test-img` examples.
2. **Display labels** — three `.names` tables mapping model output IDs to names for printing and drawing.
3. **Reference illustrations** — `result_detect*.jpg` and four `ultralytics_YOLO_*_demo` captures showing each task's result format.

<a id="files"></a>

## Directory structure

```text
test_data/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── bus.jpg  # Detection/segmentation/pose input image
├── zebra_cls.jpg  # Classification input image
├── coco_classes.names  # Display labels for detect/seg/pose
├── imagenet_classes.names  # Display labels for cls
├── ultralytics_dota_classes.names  # Display labels for obb
├── result_detect.jpg  # S detection result illustration (bus scene)
├── result_detect_yolo26.jpg  # S YOLO26 detection result illustration
├── ultralytics_YOLO_Detect_demo.jpg  # Detection example screenshot
├── ultralytics_YOLO_Pose_demo.jpg  # Pose example screenshot
├── ultralytics_YOLO_Seg_demo.jpg  # Segmentation example screenshot
└── ultralytics_YOLO_CLS_demo.png  # Classification example screenshot
```

## Files

| File | Role | Pixels |
| --- | --- | --- |
| `bus.jpg` | Default detection/segmentation/pose CLI input | 810×1080 |
| `zebra_cls.jpg` | Classification CLI input | 376×376 |
| `coco_classes.names` | Display labels for detect/seg/pose (80 entries) | — |
| `imagenet_classes.names` | Display labels for cls (1000 entries) | — |
| `ultralytics_dota_classes.names` | Display labels for obb (15 entries) | — |
| `result_detect.jpg` | S reference detection illustration (bus scene) | 810×1080 |
| `result_detect_yolo26.jpg` | S reference YOLO26 detection illustration | 810×1080 |
| `ultralytics_YOLO_Detect_demo.jpg` | Detection example screenshot, referenced by the [sample guide](../README.md#expected-results) | 2900×1888 |
| `ultralytics_YOLO_Pose_demo.jpg` | Pose example screenshot | 2898×1888 |
| `ultralytics_YOLO_Seg_demo.jpg` | Segmentation example screenshot | 2898×1888 |
| `ultralytics_YOLO_CLS_demo.png` | Classification example screenshot | 2846×1268 |

The result illustrations were produced from the bundled `bus.jpg` input; the
`ultralytics_YOLO_*_demo` files are example screenshots. Treat them as result
format references.

<a id="labels"></a>
## Label tables: display names only

The Python CLI loads these tables automatically. `main.py` defaults to
`coco_classes.names` for detect/seg/pose, `imagenet_classes.names` for cls and
`ultralytics_dota_classes.names` for obb; `--label-file` overrides the choice
for any task.

### coco_classes.names

80 non-empty lines, one display name per line; line N names model output index
N−1 (zero-based). The order is the standard Ultralytics COCO-80 output order
(index 0 `person` … index 79 `toothbrush`). Six entries use historical
VOC-style spellings instead of COCO names — index 3 `motorbike`,
4 `aeroplane`, 57 `sofa`, 58 `pottedplant`, 60 `diningtable`, 62 `tvmonitor` —
see the [COCO dataset guide](../../../../datasets/coco/README.md). Both
spellings name the same output columns; keep the file order unchanged.

### imagenet_classes.names

A Python dict literal spanning all 1000 classes, `{index: 'comma, separated,
synonyms'}`, keys 0–999 in the standard ILSVRC-2012 order (index 0
`tench, Tinca tinca`, index 340 `zebra`, index 999 `toilet tissue, toilet
paper, bathroom tissue`). The values are display names, not WordNet synset
IDs (`n########`). The loader accepts this dict form and also a
one-name-per-line list.

### ultralytics_dota_classes.names

15 lines naming the YOLO26 OBB model's 15 output columns, in that model's own
output order: `plane` 0, `ship` 1, `storage-tank` 2, `baseball-diamond` 3,
`tennis-court` 4, `basketball-court` 5, `ground-track-field` 6, `harbor` 7,
`bridge` 8, `large-vehicle` 9, `small-vehicle` 10, `helicopter` 11,
`roundabout` 12, `soccer-ball-field` 13, `swimming-pool` 14. DOTA's native
annotations have no official numeric category-ID table; this numbering is the
model output order. It differs from the
[`datasets/dotav1/dota_classes.names`](../../../../datasets/dotav1/README.md)
fixed repository listing: the two coincide only at index 0, so naming outputs
that follow one order with the other file mislabels 14 of the 15 categories.
See the [DOTA dataset guide](../../../../datasets/dotav1/README.md) for the
repository listing's order.

### Custom models and mismatch behavior

For a compiled model whose class count or order differs, pass a matching
`--label-file` (one name per line in model output order; cls additionally
accepts the dict-literal form) and, for detect/seg/obb custom graphs, the
matching `--classes-num`. A wrong order does not fail: predictions keep
their boxes and scores but every rendered or printed name is shifted.
When a class ID falls outside the loaded label list:

- The detect text report and OBB drawing print the numeric ID instead of a
  name; classification prints `Unknown(<id>)`.
- Detect/seg rendered-image drawing indexes the list directly, so the run
  stops with an `IndexError` before a result image is written.

<a id="inputs"></a>
## Using the bundled and custom images

`--test-img` accepts any image OpenCV can read as three-channel BGR; the task
resizes it to the model's input geometry (letterbox by default, with the
resize exceptions documented in the
[Python runtime guide](../runtime/python/README.md#parameters)) and all
results restore original-image pixel coordinates. Input geometry comes from
model metadata, not from the filename.

The bundled images are the documented fixed inputs. Run from the repository
root on a matching board; with no arguments at all the board is auto-detected
and yolo11 default-scale detection runs on `bus.jpg`:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py
```

Classification uses `zebra_cls.jpg`:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo26 --task cls \
  --test-img samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg --topk 5
```

For your own image, pass its path as `--test-img`. This example pairs a custom
image with an explicit local model:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

Replace `/data/image.jpg` with any readable image and the model path with your
prepared artifact (see the [model guide](../model/README.md)). To feed a
custom class list with a custom image, add `--label-file` to the same form:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img /data/image.jpg --label-file /data/my_classes.txt \
  --img-save-path /tmp/custom-detect.jpg
```

No aerial/OBB input is bundled: supply your own aerial image for obb. The
[YOLO26 OBB stages](../runtime/python/README.md#yolo26-obb-stages) use the
bundled bus image only to demonstrate calling the API.

### Output locations and console semantics

Detect/seg/pose/obb write the rendered image to `--img-save-path` — default
`result.jpg`, relative to the calling working directory — creating its parent
directory, overwriting an existing file, and printing
`[Saved] Result saved to: <path>` on success. Classification prints results
and writes no image. Exit status 0 means the command completed; an empty
detection list is a valid result. Console formats (placeholder values):

```text
Detection Report: N objects found
  [0] <class name>: <score> | Box (<x1>, <y1>, <x2>, <y2>)
```

```text
Top-K Classification Results:
  [0] <label>: <probability>
```

`<score>` and `<probability>` are model confidences after the task's own
post-processing (detection threshold/NMS filter; classification Softmax),
printed with two and four decimals respectively; boxes use original-image
pixels and zero-based class IDs. Detect prints the per-object report,
classification prints Top-K, and seg, pose and obb communicate through the
rendered image. For dataset accuracy and performance, use the
[evaluator](../evaluator/README.md).

The C++ reference programs accept the same images as positional arguments
(their [guide](../runtime/cpp/README.md) pairs them with
`test_data/bus.jpg` and `test_data/zebra_cls.jpg`), but they do not read
the `.names` files in this directory: detect/segment compile an 80-entry COCO
name array with spellings (`motorcycle`, `airplane`, `couch`,
`potted plant`, `dining table`, `tv` — six entries differ from this
directory's VOC-style synonyms, same order), pose compiles 17 COCO keypoint
names, and classification compiles the 1000-entry ImageNet order in
`runtime/cpp/inc/imagenet_labels.hpp`. Conversion post-compilation checks likewise quote
`test_data/bus.jpg` as their example input (see the
[conversion guide](../conversion/README.md#validation)).

<a id="boundaries"></a>
## Data preparation

Detection, segmentation and pose evaluation uses COCO annotation JSON.
ImageNet ground truth comes from `--val-txt` or a `--label-file` containing
synset IDs in class order. The evaluator's labels use `n########` identifiers;
runtime display names use this directory's `imagenet_classes.names`. The OBB
entry exports predictions. Prepare datasets with the
[COCO](../../../../datasets/coco/README.md),
[ImageNet](../../../../datasets/imagenet/README.md) and
[DOTA](../../../../datasets/dotav1/README.md) guides; see
[evaluator parameters](../evaluator/README.md#dataset). Calibration samples
are selected per the
[conversion calibration recipe](../conversion/README.md#calibration).
