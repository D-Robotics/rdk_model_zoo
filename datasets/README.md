English | [简体中文](./README_cn.md)

# Dataset Resources

Shared dataset-side resources for the samples in this checkout: class/label
lists, a small number of bundled example images, and one acquisition script.
This directory does **not** contain full datasets; downloads are kept out of
Git only where ignore rules cover the layout — the COCO script's default
`coco_full/` output is **not** ignored, so run it outside the checkout or add a
local exclusion (see the [COCO guide](coco/README.md)). Every dataset is
acquired manually under its own terms.

<a id="index"></a>
## Index

| Directory | Bundled resources | Used by | Guide |
| --- | --- | --- | --- |
| [coco/](coco/) | `coco_classes.names` (80 classes), example images `bus.jpg` / `kite.jpg`, full-dataset download script | [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README.md) detection/segmentation/pose evaluator, [YOLOE](../samples/vision/yoloe/README.md) evaluator (COCO-format input), YOLOv5 sample (test images) | [COCO guide](coco/README.md) |
| [imagenet/](imagenet/) | `imagenet_classes.names` (1000-class dict literal), example image `zebra_cls.jpg` | All classification samples (`--label-file`), [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README.md) classification evaluator (Top-1/Top-5) | [ImageNet guide](imagenet/README.md) |
| [dotav1/](dotav1/) | `dota_classes.names` (15 classes, fixed repository listing), three example tiles | No current unified script reads this directory; the archived X5 YOLO26 OBB flow did | [DOTA guide](dotav1/README.md) |
| [PascalVOC/](PascalVOC/) | Reference links only — no bundled files | [UNet](../samples/vision/unet/README.md) evaluator (VOC 2012 segmentation, acquired separately) | [Pascal VOC guide](PascalVOC/README.md) |
| [yoloe/](yoloe/) | `yoloe_seg_pf_classes.names` — fixed 4585-class prompt-free vocabulary | [YOLOE](../samples/vision/yoloe/README.md) sample and evaluator | [YOLOE guide](yoloe/README.md) |

<a id="boundaries"></a>
## Boundaries

- **No dataset downloads are automated here except COCO.** Only
  [coco/download_full_coco.sh](coco/download_full_coco.sh) exists; ImageNet,
  DOTA and Pascal VOC have no downloader in this repository and must be
  acquired manually from their official sources. No script in this directory
  runs as part of host tests, and none was executed to write these guides.
- **Licenses and terms are per dataset.** Obtain each dataset under its own
  terms; the guides link the official sites. Bundled label lists and example
  images are small excerpts kept for offline smoke checks, not redistributions
  of the datasets.
- **Class index ≠ dataset category ID.** Model output columns are contiguous
  indices. COCO annotation files use sparse numeric IDs; DOTA's native
  annotations carry category **names**, so any numeric DOTA ID space belongs to
  the conversion that defines it, not to the format itself. The per-dataset
  guides state the exact mapping; the [YOLOE guide](yoloe/README.md) covers the
  separate 4585-class prompt-free vocabulary, whose indices are neither COCO
  category IDs nor Ultralytics COCO-80 indices.

<a id="provenance"></a>
## Provenance

The dataset resources in this tree (label lists, example images, the COCO
download script) are byte-identical to the X5 delivery branch snapshot
(`ac11571`); the README guides were rewritten for this unified tree while the
resource files were left untouched. The S delivery line (`380e1a2`) carries the
same COCO, DOTA and ImageNet resources (its `coco_classes.names` differs only
in CRLF line endings) and no `PascalVOC/` or `yoloe/` directories. Frozen per-platform
copies remain under `platforms/x5/datasets/` and `platforms/s/datasets/` as
historical references; they are archived snapshots, not the active entry
points. The unified sample documentation lives beside each sample under
`samples/`; the active manifests are `docs/release/x5/models.yaml` and
`docs/release/s/models.yaml`.
