English | [简体中文](./README_cn.md)

# Dataset Resources

Shared dataset-side resources include class/label lists, example images and
the COCO acquisition script. Full datasets are acquired separately under their
own terms. The COCO script writes `coco_full/` under its current working
directory; choose a data directory outside the repository when downloading.

<a id="index"></a>
## Index

| Directory | Bundled resources | Used by | Guide |
| --- | --- | --- | --- |
| [coco/](coco/) | `coco_classes.names` (80 classes), example images `bus.jpg` / `kite.jpg`, full-dataset download script | [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README.md) detection/segmentation/pose evaluator, [YOLOE](../samples/vision/yoloe/README.md) evaluator (COCO-format input), YOLOv5 sample (test images) | [COCO guide](coco/README.md) |
| [imagenet/](imagenet/) | `imagenet_classes.names` (1000-class dict literal), example image `zebra_cls.jpg` | All classification samples (`--label-file`), [Ultralytics YOLO](../samples/vision/ultralytics_yolo/README.md) classification evaluator (Top-1/Top-5) | [ImageNet guide](imagenet/README.md) |
| [dotav1/](dotav1/) | `dota_classes.names` (15 classes, fixed repository listing), three example tiles | DOTA category mapping and example tiles; Ultralytics YOLO OBB uses its sample-local model-order label file | [DOTA guide](dotav1/README.md) |
| [PascalVOC/](PascalVOC/) | Reference links only — no bundled files | [UNet](../samples/vision/unet/README.md) evaluator (VOC 2012 segmentation, acquired separately) | [Pascal VOC guide](PascalVOC/README.md) |
| [yoloe/](yoloe/) | `yoloe_seg_pf_classes.names` — fixed 4585-class prompt-free vocabulary | [YOLOE](../samples/vision/yoloe/README.md) sample and evaluator | [YOLOE guide](yoloe/README.md) |

<a id="boundaries"></a>
## Boundaries

- **Only COCO has a download script here.**
  [coco/download_full_coco.sh](coco/download_full_coco.sh) prepares COCO 2017;
  obtain ImageNet, DOTA and Pascal VOC from their official sources.
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

Sample-specific data preparation and input conventions are documented beside each implementation under `samples/`. Published artifact inventories are `docs/release/x5/models.yaml` and `docs/release/s/models.yaml`.
