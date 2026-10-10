English | [简体中文](README_cn.md)

# Ultralytics YOLO test data

This directory provides eleven fixed files: input images, runtime display labels
and reference result illustrations.

1. **Input images** — `bus.jpg` and `zebra_cls.jpg`, used by the runtime, C++ and conversion `--test-img` examples.
2. **Display labels** — three `.names` tables mapping model output IDs to names for printing and drawing.
3. **Reference illustrations** — `result_detect*.jpg` and four `ultralytics_YOLO_*_demo` captures showing each task's result format.

<a id="files"></a>

## Directory structure

```text
test_data/
├── README.md  # English instructions
├── README_cn.md  # Chinese instructions
├── coco_classes.names  # Source or data file
├── imagenet_classes.names  # Source or data file
└── ultralytics_dota_classes.names  # Source or data file
```

## Files and byte identities

| File | Role | Pixels | SHA-256 |
| --- | --- | --- | --- |
| `bus.jpg` | Default detection/segmentation/pose CLI input | 810×1080 | `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` |
| `zebra_cls.jpg` | Classification CLI input | 376×376 | `53c9f26d927b507fb3b9b68005fd8dd3ba329a0528f9c1f51acbde55e9525462` |
| `coco_classes.names` | Display labels for detect/seg/pose (80 entries) | — | `634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84` |
| `imagenet_classes.names` | Display labels for cls (1000 entries) | — | `e6ac1e05778e809de37a089105170398e5d0b5f7e337165c94aeaacb80fbba14` |
| `ultralytics_dota_classes.names` | Display labels for obb (15 entries) | — | `a6c8b62b2dae0ddc151a4cbae52b8db84542e2ca0e254e5214d86be9d180b6b9` |
| `result_detect.jpg` | S reference detection illustration (bus scene) | 810×1080 | `5d792a474744924eb31b37e2f0d888931955151785cfba953104549462afa218` |
| `result_detect_yolo26.jpg` | S reference `ultralytics_yolo26` detection illustration | 810×1080 | `2631c66105d3e65373db1b60e2f968e4a54663b039646f38bfcf2355d53375c3` |
| `ultralytics_YOLO_Detect_demo.jpg` | Detection reference capture, referenced by the [sample guide](../README.md#expected-results) | 2900×1888 | `926ad7e306cc471adf5d3b3e237199cd02d1b4648a96c1a214d984e9747cba6f` |
| `ultralytics_YOLO_Pose_demo.jpg` | Pose reference capture, reference result format | 2898×1888 | `194ffd24552ae45ae6a526248fc7293f8e44de0bb209b5647f2016c67f782b1c` |
| `ultralytics_YOLO_Seg_demo.jpg` | Segmentation reference capture, reference result format | 2898×1888 | `a80979ccab290ec5f56116ab554fc1ce1ceed1f0cabcb139371611dab4507731` |
| `ultralytics_YOLO_CLS_demo.png` | Classification reference capture, reference result format | 2846×1268 | `7ea5f2288ad274dc3666b2b8351414dbeb48787348fc7ec272299a0c6b61329f` |

The observed SHA-256 digests identify the bundled file contents for checksum
comparison and detection of byte changes.

Byte-identity relations verified inside this checkout:

- `bus.jpg` is byte-identical to [`datasets/coco/assets/bus.jpg`](../../../../datasets/coco/README.md)
  and to the archived delivery copies under
  `platforms/x5/samples/vision/ultralytics_yolo/test_data/` and
  `platforms/s/samples/vision/ultralytics_yolo/test_data/`.
- `zebra_cls.jpg` is byte-identical to
  [`datasets/imagenet/asset/zebra_cls.jpg`](../../../../datasets/imagenet/README.md)
  and `samples/vision/resnet/test_data/zebra_cls.jpg`.
- `coco_classes.names` is byte-identical to
  [`datasets/coco/coco_classes.names`](../../../../datasets/coco/README.md);
  `imagenet_classes.names` is byte-identical to
  [`datasets/imagenet/imagenet_classes.names`](../../../../datasets/imagenet/README.md).
  There is only one content set per vocabulary in this checkout.
- `result_detect.jpg` is byte-identical to the file of the same name in the
  archived S delivery (`platforms/s/samples/vision/ultralytics_yolo/test_data/`);
  `result_detect_yolo26.jpg` is the S `ultralytics_yolo26` delivery's
  `result_detect.jpg`, carried under a distinct filename in this directory. The
  two illustrations have different bytes and different on-image labels; both
  are kept with distinct identities.

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
spellings name the same output columns; do not renumber the file.

### imagenet_classes.names

A Python dict literal spanning all 1000 classes, `{index: 'comma, separated,
synonyms'}`, keys 0–999 in the standard ILSVRC-2012 order (index 0
`tench, Tinca tinca`, index 340 `zebra`, index 999 `toilet tissue, toilet
paper, bathroom tissue`); the file has no trailing newline. The values are
human-readable display names, not WordNet synset IDs (`n########`), and cannot
substitute for them. The loader accepts this dict form and also a
one-name-per-line list.

### ultralytics_dota_classes.names

15 lines naming the YOLO26 OBB model's 15 output columns, in that model's own
output order: `plane` 0, `ship` 1, `storage-tank` 2, `baseball-diamond` 3,
`tennis-court` 4, `basketball-court` 5, `ground-track-field` 6, `harbor` 7,
`bridge` 8, `large-vehicle` 9, `small-vehicle` 10, `helicopter` 11,
`roundabout` 12, `soccer-ball-field` 13, `swimming-pool` 14. DOTA's native
annotations have no official numeric category-ID table; this numbering is the
model output order, nothing else. It is **not** the same order as
[`datasets/dotav1/dota_classes.names`](../../../../datasets/dotav1/README.md)
(the fixed repository listing): the two coincide only at index 0. Using one
file to name outputs that follow the other silently mislabels 14 of the 15
categories. See the [DOTA dataset guide](../../../../datasets/dotav1/README.md)
for the full ordering warning.

### Custom models and mismatch behavior

For a compiled model whose class count or order differs, pass a matching
`--label-file` (one name per line in model output order; cls additionally
accepts the dict-literal form) and, for detect/seg/obb custom graphs, the
matching `--classes-num`. A wrong order does not fail loudly: predictions keep
their boxes and scores but every rendered or printed name is shifted.
Observed behavior when a class ID falls outside the loaded label list:

- The detect text report prints the numeric ID instead of a name.
- OBB drawing prints the numeric ID; classification prints `Unknown(<id>)`.
- Detect/seg rendered-image drawing indexes the list directly, so the run
  stops with an `IndexError` before a result image is written.

<a id="inputs"></a>
## Using the bundled and custom images

`--test-img` accepts any image OpenCV can read as three-channel BGR; the task
resizes it to the model's input geometry (letterbox by default, with the
resize exceptions documented in the
[Python runtime guide](../runtime/python/README.md#parameters)) and all
results restore original-image pixel coordinates. A filename is never an
input override: model metadata decides input geometry.

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

For your own image, pass its path as `--test-img`; the following existing
example form from the runtime guide pairs a custom image with an explicit
local model:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform s600 --family yolo11 --task detect \
  --model-path /models/yolo11n_nashp_640x640_nv12.hbm \
  --test-img /data/image.jpg --img-save-path /tmp/yolo-s600.jpg
```

Replace `/data/image.jpg` with any readable image; replace the model path with
your prepared artifact (see the [model guide](../model/README.md)). To feed a
custom class list with a custom image, add `--label-file` to the same form:

```bash
python samples/vision/ultralytics_yolo/runtime/python/main.py \
  --platform x5 --family yolov8 --task detect \
  --test-img /data/image.jpg --label-file /data/my_classes.txt \
  --img-save-path /tmp/custom-detect.jpg
```

No aerial/OBB input is bundled: supply your own aerial image for obb, and do
not treat `bus.jpg` OBB output as an aerial-detection expectation — the
[YOLO26 OBB stages](../runtime/python/README.md#yolo26-obb-stages) use the
bundled bus image only to demonstrate calling the API.

### Output locations and console semantics

Detect/seg/pose/obb write the rendered image to `--img-save-path` — default
`result.jpg`, relative to the calling working directory — creating its parent
directory, overwriting an existing file, and printing
`[Saved] Result saved to: <path>` on success. Classification prints results
and writes no image. Exit status 0 means the command completed; an empty
detection list is a valid result. Console formats, with placeholder values —
these are format templates, not measurements:

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
pixels and zero-based class IDs. Apart from the model/protocol information the
runtime prints for every task, detect prints the per-object report and
classification prints Top-K; seg, pose and obb communicate through the
rendered image. One rendered image is not dataset accuracy or performance
evidence — use the [evaluator](../evaluator/README.md) for that.

The C++ reference programs accept the same images as positional arguments
(their [guide](../runtime/cpp/README.md) pairs them with
`test_data/bus.jpg` and `test_data/zebra_cls.jpg`), but they do **not** read
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
[evaluator parameters](../evaluator/README.md#dataset).

Select representative dataset samples using the
[conversion calibration recipe](../conversion/README.md#calibration).
Use a new output path for inference and compare files written by that successful
run. Host regression tests construct synthetic inputs under `../tests/`.

<a id="provenance"></a>
## Provenance

The result illustrations use the bundled `bus.jpg` input. Use a fresh `--img-save-path` when producing your own detection, pose, segmentation or classification output.

The `ultralytics_YOLO_*_demo` files are example screenshots. Use the current runtime parameter table and command blocks for paths and thresholds.

Dataset scoring requires the labeled validation inputs described in the [evaluator guide](../evaluator/README.md).
