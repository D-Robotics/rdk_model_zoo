English | [简体中文](./README_cn.md)

# DOTA-v1.0 Dataset Resources

**DOTA-v1.0** is a large-scale aerial-image benchmark for oriented object
detection: 2,806 large aerial images annotated with oriented bounding boxes
over 15 categories (see the official page for the current figures and split
definitions). This directory bundles the 15-class list and three example
tiles; the dataset itself is **not included** and there is **no download
script** — acquisition is manual from the official source under its terms.

<a id="files"></a>
## Bundled files

```text
dotav1/
├── README.md             # this guide
├── README_cn.md          # Chinese guide
├── dota_classes.names    # 15 class names, one per line
└── asset/
    ├── P0009.png         # example tile (original DOTA image ID)
    ├── P0014.png
    └── P0035.png
```

### dota_classes.names

15 entries verified from the file, one class name per line. DOTA's native
annotations record each instance as **eight coordinates, a category name and a
difficulty flag** (official annotation specification); there is **no official
numeric category-ID table**. This file is therefore a fixed repository-side
listing: its line order is simply the order this checkout uses wherever a
numbered DOTA category list is needed. Any tool that converts DOTA into a
numeric-ID format (for example a COCO-style conversion) defines its own index
mapping — obtain and state that mapping explicitly when scoring; do not assume
it matches this listing or the model-output order below.

```text
plane, baseball-diamond, bridge, ground-track-field, small-vehicle,
large-vehicle, ship, tennis-court, basketball-court, storage-tank,
soccer-ball-field, roundabout, harbor, swimming-pool, helicopter
```

<a id="ordering"></a>
## Two local orderings you must not mix up

Two different 15-class orders exist in this checkout, and they are **not
interchangeable**:

| File | Order | Meaning |
| --- | --- | --- |
| `datasets/dotav1/dota_classes.names` (this directory) | fixed repository listing (see above) | Line positions used wherever this checkout needs a numbered DOTA category list; not an official ID table |
| `samples/vision/ultralytics_yolo/test_data/ultralytics_dota_classes.names` | Ultralytics DOTA model output order (e.g. `ship` = 1, `storage-tank` = 2, 0-based) | Names model output columns of the Ultralytics OBB heads |

The two orders coincide only at index 0 (`plane`); the other 14 positions
differ, so using one file to name outputs or conversions that follow the other
silently mislabels those categories. When you score against a DOTA dataset
converted to numeric IDs, the conversion owns the ID mapping — require it
explicitly instead of assuming either local order. The current OBB path is
also prediction-only: the
[Ultralytics YOLO OBB evaluator](../../samples/vision/ultralytics_yolo/evaluator/README.md)
exports rotated rectangles/polygons and explicitly computes **no DOTA AP**, and
its `--label-path` option exists only for legacy command compatibility and is
not scored. The prediction export is the evaluation interface: DOTA AP is
computed by an external DOTA scorer from the exported predictions, so the
exports themselves are not accuracy results.

<a id="usage"></a>
## Where these resources are used

Use `asset/P0009.png`, `asset/P0014.png` and `asset/P0035.png` as OBB example inputs. The [Ultralytics YOLO sample](../../samples/vision/ultralytics_yolo/README.md) uses `test_data/ultralytics_dota_classes.names` to name its model output columns.

For Ultralytics OBB, choose the sample-local model-order label file; for a converted evaluation dataset, provide the category mapping produced by that conversion.

The three `asset/` tiles are example excerpts for visual reference and offline
experiments only — not a usable evaluation split.

<a id="reference"></a>
## Official site and terms

- Official page: <https://captain-whu.github.io/DOTA/dataset.html>
  (download application, splits and terms are published there)

Dataset and annotation specification: [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html). Use an explicit category-name/index mapping when converting the annotations for evaluation.
