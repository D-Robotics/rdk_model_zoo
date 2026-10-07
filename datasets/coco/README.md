English | [简体中文](./README_cn.md)

# COCO Dataset Resources

**COCO (Common Objects in Context)** is one of the most widely used public
datasets in computer vision, serving object detection, instance segmentation,
keypoint detection and general image understanding. Its complex scenes, rich
category set and real-world context make it the standard benchmark for model
evaluation. This directory bundles small offline examples plus the full-dataset
download script for development, debugging and validation.

The dataset itself is **not included** in Git; acquisition is manual and must
follow the official terms.

<a id="files"></a>
## Bundled files

```text
coco/
├── README.md                  # this guide
├── README_cn.md               # Chinese guide
├── download_full_coco.sh      # full COCO 2017 download script
├── coco_classes.names         # 80 class names, one per line
└── assets/
    ├── bus.jpg                # example image (byte-identical to the sample test_data copy)
    └── kite.jpg               # example image (byte-identical to the YOLOv5 sample copy)
```

### coco_classes.names

80 entries verified from the file, one class name per line. Line position is
the model output class index (0–79). The index order matches the standard
Ultralytics COCO-80 output order; six display names use VOC-style synonyms
that differ from the canonical COCO names:

| Index | This file | Canonical COCO name |
| --- | --- | --- |
| 3 | `motorbike` | motorcycle |
| 4 | `aeroplane` | airplane |
| 57 | `sofa` | couch |
| 58 | `pottedplant` | potted plant |
| 60 | `diningtable` | dining table |
| 62 | `tvmonitor` | tv |

### assets/

`bus.jpg` and `kite.jpg` are the two bundled example photos. They are byte
identical to the copies used as default test images by
[Ultralytics YOLO](../../samples/vision/ultralytics_yolo/README.md)
(`test_data/bus.jpg`) and
[YOLOv5](../../samples/vision/yolov5/README.md) (`test_data/bus.jpg` for X5,
`test_data/kite.jpg` for S targets). They enable offline runtime smoke checks
without any dataset download.

### Class index vs COCO category ID

This is the distinction the evaluators depend on:

- **Model output index**: contiguous 0–79, in `coco_classes.names` line order.
- **COCO annotation `category_id`**: sparse IDs 1–90; ten IDs (12, 26, 29, 30,
  45, 66, 68, 69, 71, 83) do not exist. Index 0 → category 1 (person), index 79
  → category 90 (toothbrush).

The [Ultralytics YOLO evaluator](../../samples/vision/ultralytics_yolo/evaluator/README.md)
applies the standard index→ID mapping (`COCO_CATEGORY_IDS` in
`eval_common.py`) regardless of which category subset an annotation file
contains. Never write `coco_classes.names` line numbers into COCO JSON as
`category_id`.

<a id="download"></a>
## Full COCO download script

[download_full_coco.sh](download_full_coco.sh) fetches and unpacks the official
COCO 2017 images and annotations.

Behavior, read from the script:

- Downloads three archives with `wget -c --no-check-certificate` from
  `images.cocodataset.org`: `zips/train2017.zip`, `zips/val2017.zip` and
  `annotations/annotations_trainval2017.zip`. `wget -c` resumes partial
  downloads when re-run.
- **Output directory is `coco_full/` relative to the current working
  directory**, not a fixed location. Run it from the directory where you want
  the data, for example `datasets/coco/`. The script is checked in without the
  executable bit, so invoke it through `bash`:

  ```bash
  # cwd: /data/coco
  bash /path/to/rdk_model_zoo/datasets/coco/download_full_coco.sh
  ```

- Extracts each archive with `unzip -q` inside `coco_full/`, then deletes the
  three `.zip` files. Requires `bash`, `wget` and `unzip`.
- Resulting layout (train2017 has 118,287 images, val2017 has 5,000;
  the annotations archive carries instances/captions/person_keypoints JSON for
  both splits):

  ```text
  coco_full/
  ├── train2017/            # training images
  ├── val2017/              # validation images
  └── annotations/
      ├── instances_train2017.json
      ├── instances_val2017.json
      ├── captions_train2017.json
      ├── captions_val2017.json
      ├── person_keypoints_train2017.json
      └── person_keypoints_val2017.json
  ```

- `test2017` is not downloaded. Bandwidth is the main cost; if the official
  source is slow, the script may be pointed at a mirror by editing the URLs —
  verify any mirror's integrity yourself before use.

Keep downloaded data outside the repository. The script's `coco_full/` output
is relative to the current working directory; the command above runs it from
`/data/coco`.

<a id="usage"></a>
## Where these resources are used

| Consumer | Use |
| --- | --- |
| [Ultralytics YOLO evaluator](../../samples/vision/ultralytics_yolo/evaluator/README.md) | Detection/segmentation use `instances_val2017.json`; pose uses `person_keypoints_val2017.json`; commands run from the repository root with your prepared image/annotation paths |
| [YOLOE evaluator](../../samples/vision/yoloe/evaluator/README.md) | Any COCO-format instance dataset (images + categories + annotations), not necessarily COCO itself |
| [YOLOv5 sample](../../samples/vision/yolov5/README.md) | `assets/bus.jpg` / `assets/kite.jpg` as default test images |
| Classification samples | Not used; they use [ImageNet](../imagenet/README.md) labels |

For DOTA oriented-box evaluation see [dotav1](../dotav1/README.md); the
current OBB evaluator exports predictions only and computes no COCO-style AP
over DOTA.

<a id="reference"></a>
## Official site and terms

- Official site: <https://cocodataset.org>
- Images originate from Flickr and remain under their respective terms of use;
  annotations are published under Creative Commons Attribution 4.0. Confirm
  current terms on the official site before redistribution or publication.

Dataset and annotation source: [COCO](https://cocodataset.org/). The download script prepares COCO 2017 train/val images and annotations; the bundled display labels use the model’s contiguous 80-class order.
