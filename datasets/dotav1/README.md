English | [简体中文](./README_cn.md)

> Historical `platforms/` paths below name the pre-unification trees, removed from the active branch on 2026-10-01. Read them from the pinned commit `d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d` (for example `git show d2d2a4e0:<path>`, or a temporary `git worktree add <dir> d2d2a4e0a898697bdfe5f68a9740a8c7d7cad57d`); see `docs/migration/2026-09-30-model-examples.md`.


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
not scored. No DOTA scorer is implemented in this checkout; do not present
OBB prediction exports as an accuracy result.

<a id="usage"></a>
## Where these resources are used

No script or evaluator in the unified `samples/` tree reads this directory
today. Historical consumers, preserved as archived provenance:

- The X5 delivery branch's `ultralytics_yolo26` sample used
  `asset/P0009.png` as an OBB test image and `dota_classes.names` as its label
  file — see the archived X5 YOLO26 runtime guide (historical `../../platforms/x5/samples/vision/ultralytics_yolo26/runtime/python/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md)
  and evaluator guide (historical `../../platforms/x5/samples/vision/ultralytics_yolo26/evaluator/README.md` at pinned commit `d2d2a4e0`; see docs/migration/2026-09-30-model-examples.md).
- The current [Ultralytics YOLO sample](../../samples/vision/ultralytics_yolo/README.md)
  keeps its own `test_data/ultralytics_dota_classes.names` (model order) and
  bundled OBB test images; the archived files above remain frozen references.

The three `asset/` tiles are example excerpts for visual reference and offline
experiments only — not a usable evaluation split.

<a id="reference"></a>
## Official site and terms

- Official page: <https://captain-whu.github.io/DOTA/dataset.html>
  (download application, splits and terms are published there)

Inherited source: the bare link README from the X5 (`ac11571`) and S
(`380e1a2`) delivery lines is expanded here with the verified file inventory
and ordering warnings; the archived copies remain under
`platforms/x5/datasets/dotav1/` and `platforms/s/datasets/dotav1/`.
