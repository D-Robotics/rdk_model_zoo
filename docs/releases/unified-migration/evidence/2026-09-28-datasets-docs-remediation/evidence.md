# H8 datasets docs remediation — static evidence (2026-09-28)

## 1. Scope of written files

```
 M datasets/PascalVOC/README.md
 M datasets/PascalVOC/README_cn.md
 M datasets/coco/README.md
 M datasets/dotav1/README.md
 M datasets/imagenet/README.md
 M datasets/imagenet/README_cn.md
 M datasets/yoloe/README.md
 M datasets/yoloe/README_cn.md
 M samples/vision/ultralytics_yolo/evaluator/README.md
 M samples/vision/ultralytics_yolo/evaluator/README_cn.md
 M samples/vision/yoloe/evaluator/README.md
 M samples/vision/yoloe/evaluator/README_cn.md
?? datasets/README.md
?? datasets/README_cn.md
?? datasets/coco/README_cn.md
?? datasets/dotav1/README_cn.md
```

## 2. Dataset resource inventory (read from actual files)
```
      80 datasets/coco/coco_classes.names
      15 datasets/dotav1/dota_classes.names
     999 datasets/imagenet/imagenet_classes.names
    4585 datasets/yoloe/yoloe_seg_pf_classes.names
    5679 total
imagenet dict entries: 1000, contiguous 0..999: True
imagenet[0]='tench, Tinca tinca'; imagenet[340]='zebra'; imagenet[999]='toilet tissue, toilet paper, bathroom tissue'

634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84  datasets/coco/coco_classes.names
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63  datasets/coco/assets/bus.jpg
87410f52f79cc256c3949b48a3548632c07c8e9943468ef8caa6250f6643be8e  datasets/coco/assets/kite.jpg
6eb0bc339dd1fbe61c0a2e7dbf40da7f9324c19a818628ff35454590dee23cb7  datasets/dotav1/dota_classes.names
bd2ae0bc2e3e13d5684b153bd3d44f817964829a1c01e92e2666cdd662d4eba3  datasets/dotav1/asset/P0009.png
6a5f1743424d7f0d17883cc9fd38a69f49c6aa6965f4352635233394e0bd2068  datasets/dotav1/asset/P0014.png
dde925501ff0f2bddb7e28198fdd0586620f7a7ef587412717f666b7ea6584c9  datasets/dotav1/asset/P0035.png
e6ac1e05778e809de37a089105170398e5d0b5f7e337165c94aeaacb80fbba14  datasets/imagenet/imagenet_classes.names
53c9f26d927b507fb3b9b68005fd8dd3ba329a0528f9c1f51acbde55e9525462  datasets/imagenet/asset/zebra_cls.jpg
1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3  datasets/yoloe/yoloe_seg_pf_classes.names
1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3  samples/vision/yoloe/test_data/classes.names
634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84  samples/vision/ultralytics_yolo/test_data/coco_classes.names
e6ac1e05778e809de37a089105170398e5d0b5f7e337165c94aeaacb80fbba14  samples/vision/ultralytics_yolo/test_data/imagenet_classes.names
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63  samples/vision/ultralytics_yolo/test_data/bus.jpg
53c9f26d927b507fb3b9b68005fd8dd3ba329a0528f9c1f51acbde55e9525462  samples/vision/ultralytics_yolo/test_data/zebra_cls.jpg
c2e572e0b7c4c761a113efc4be6be43f78cf5c61f46979073bccc1a8ca8b1e57  samples/vision/yolov5/test_data/coco_classes.names
c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63  samples/vision/yolov5/test_data/bus.jpg
87410f52f79cc256c3949b48a3548632c07c8e9943468ef8caa6250f6643be8e  samples/vision/yolov5/test_data/kite.jpg
a6c8b62b2dae0ddc151a4cbae52b8db84542e2ca0e254e5214d86be9d180b6b9  samples/vision/ultralytics_yolo/test_data/ultralytics_dota_classes.names
53c9f26d927b507fb3b9b68005fd8dd3ba329a0528f9c1f51acbde55e9525462  samples/vision/resnet/test_data/zebra_cls.jpg
```

## 3. Source pin comparisons (read-only)
```
--- datasets tree vs X5 pin ac11571 (resources only; READMEs rewritten this round) ---
(empty = byte-identical)
--- S pin 380e1a2 vocabulary identity ---
1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3  -
1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3  -
--- S pin coco_classes.names vs current (CRLF only) ---
IDENTICAL modulo CRLF
--- archived X5 datasets/yoloe README (original source of rewritten pair) ---
pin referenced, not modified
```

## 4. DOTA ordering evidence (two different orders)
```
--- datasets/dotav1/dota_classes.names (fixed repository listing order; native DOTA annotations record category names per the official page — no official numeric ID table; corrected from "official annotation order, category IDs 1..15" per DATASET-R1) ---
     1	plane
     2	baseball-diamond
     3	bridge
     4	ground-track-field
     5	small-vehicle
     6	large-vehicle
     7	ship
     8	tennis-court
     9	basketball-court
    10	storage-tank
    11	soccer-ball-field
    12	roundabout
    13	harbor
    14	swimming-pool
    15	helicopter
--- samples/vision/ultralytics_yolo/test_data/ultralytics_dota_classes.names (model output order, 0-based) ---
     1	plane
     2	ship
     3	storage-tank
     4	baseball-diamond
     5	tennis-court
     6	basketball-court
     7	ground-track-field
     8	harbor
     9	bridge
    10	large-vehicle
    11	small-vehicle
    12	helicopter
    13	roundabout
    14	soccer-ball-field
    15	swimming-pool
--- evaluator standard COCO index->category mapping (eval_common.py COCO_CATEGORY_IDS) ---
#: Standard COCO model class-index to category-ID mapping.
COCO_CATEGORY_IDS: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
    89, 90,
)
```

## 5. YOLOE vocabulary position evidence
```
line 1 (index 0): 3D CG rendering
line 822 (index 821): chair
line 2164 (index 2163): person
total lines: 4585
--- canonical conversion hash gate (prepare.py --names requirement) ---
132:            or sha256_file(root / "source/classes.names") != graph["names_sha256"]
```

## 6. COCO download script behavior (read from script; not executed)
```
5:COCO_DIR="coco_full"
7:echo "Creating COCO dataset directory: ${COCO_DIR}"
8:mkdir -p "${COCO_DIR}"
9:cd "${COCO_DIR}"
14:wget -c --no-check-certificate https://images.cocodataset.org/zips/train2017.zip
15:wget -c --no-check-certificate https://images.cocodataset.org/zips/val2017.zip
18:wget -c --no-check-certificate https://images.cocodataset.org/annotations/annotations_trainval2017.zip
24:unzip -q train2017.zip
25:unzip -q val2017.zip
26:unzip -q annotations_trainval2017.zip
31:rm -f train2017.zip val2017.zip annotations_trainval2017.zip
33:echo "COCO dataset is ready in ${COCO_DIR}"
100644 3c9af14d83b50f208a11baea8849a02e89c7c5db 0	datasets/coco/download_full_coco.sh
(100644 = no executable bit; guides document 'bash download_full_coco.sh')
```

## 7. Static doc checks (this remediation)
```
OK: all local links and anchors resolve (16 files)
OK: anchors, heading counts and code-block counts match in all 6 pairs
OK: executable command lines byte-identical in all pairs
git diff --check: clean
```

## 8. Affected sample contract checkers
```
sample: samples/vision/ultralytics_yolo
  skipped R-STAGE-PURITY samples/vision/ultralytics_yolo/runtime/python/legacy.py: policy skip: documented compatibility shim (inference-contract §4)
  skipped R-STAGE-PURITY samples/vision/ultralytics_yolo/runtime/python/main.py: policy skip: CLI layer: saving output and argument handling live here
summary: 1 samples, 0 violations, 2 skips, 0 exemptions applied
rc=0
sample: samples/vision/yoloe
  skipped R-STAGE-PURITY samples/vision/yoloe/runtime/python/main.py: policy skip: CLI layer: saving output and argument handling live here
summary: 1 samples, 0 violations, 1 skips, 0 exemptions applied
rc=0
```

Check scripts: `check_links.py`, `check_parity.py`, `check_cmds2.py` (copied beside this file; run from the repository root).

## 9. Evaluator navigation-sentence diffs (the only sample-side edits)
```diff
diff --git a/samples/vision/ultralytics_yolo/evaluator/README.md b/samples/vision/ultralytics_yolo/evaluator/README.md
index ab8c428e..22bda47b 100644
--- a/samples/vision/ultralytics_yolo/evaluator/README.md
+++ b/samples/vision/ultralytics_yolo/evaluator/README.md
@@ -7,7 +7,7 @@ Evaluate a compiled model using the same task implementations as [the Python run
 <a id="dataset"></a>
 ## Prepare the dataset
 
-Use a matching validation split and preserve the original class order. Dataset acquisition and preparation are documented in [X5 COCO](../../../../platforms/x5/datasets/coco/README.md), [S COCO](../../../../platforms/s/datasets/coco/README.md), [X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README.md) and [S ImageNet](../../../../platforms/s/datasets/imagenet/README.md). Obtain datasets under their own licenses; they are not included in this checkout.
+Use a matching validation split and preserve the original class order. Dataset acquisition and preparation are documented in the unified [COCO](../../../../datasets/coco/README.md) and [ImageNet](../../../../datasets/imagenet/README.md) guides; the archived [X5 COCO](../../../../platforms/x5/datasets/coco/README.md), [S COCO](../../../../platforms/s/datasets/coco/README.md), [X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README.md) and [S ImageNet](../../../../platforms/s/datasets/imagenet/README.md) snapshots remain as provenance. Obtain datasets under their own licenses; they are not included in this checkout.
 
 The commands below assume this local layout; replace `/data` and `/models` with your prepared paths:
 
diff --git a/samples/vision/ultralytics_yolo/evaluator/README_cn.md b/samples/vision/ultralytics_yolo/evaluator/README_cn.md
index 12011993..8ed7a7a1 100644
--- a/samples/vision/ultralytics_yolo/evaluator/README_cn.md
+++ b/samples/vision/ultralytics_yolo/evaluator/README_cn.md
@@ -7,7 +7,7 @@
 <a id="dataset"></a>
 ## 数据集准备
 
-准备与模型类别顺序一致的验证集。获取和整理方法见[X5 COCO（英文）](../../../../platforms/x5/datasets/coco/README.md)、[S COCO（英文）](../../../../platforms/s/datasets/coco/README.md)、[X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README_cn.md)、[S ImageNet](../../../../platforms/s/datasets/imagenet/README_cn.md)。数据集不随仓库分发，使用时遵守各自许可。
+准备与模型类别顺序一致的验证集。获取和整理方法见统一的[COCO](../../../../datasets/coco/README_cn.md)与[ImageNet](../../../../datasets/imagenet/README_cn.md)指南；X5/S 平台快照保留作溯源（[X5 COCO](../../../../platforms/x5/datasets/coco/README.md)、[S COCO](../../../../platforms/s/datasets/coco/README.md)、[X5 ImageNet](../../../../platforms/x5/datasets/imagenet/README_cn.md)、[S ImageNet](../../../../platforms/s/datasets/imagenet/README_cn.md)）。数据集不随仓库分发，使用时遵守各自许可。
 
 以下示例约定本地目录如下；请将`/data`、`/models`替换为实际准备路径：
 
diff --git a/samples/vision/yoloe/evaluator/README.md b/samples/vision/yoloe/evaluator/README.md
index 43621141..c6ee3f27 100644
--- a/samples/vision/yoloe/evaluator/README.md
+++ b/samples/vision/yoloe/evaluator/README.md
@@ -7,7 +7,7 @@ Evaluate a local floating-output model with the same postprocessing used by [the
 <a id="dataset"></a>
 ## Dataset and category mapping
 
-Use a held-out COCO-format instance dataset, with `images`, `categories` and `annotations`. Each image needs integer `id`, relative `file_name`, `width` and `height`; each category needs integer `id` and `name`. Box/mask scoring needs valid instance ground truth, including bounding boxes, segmentation, area and crowd flags. Predictions-only mode accepts an image/category manifest with an empty annotations list. Dataset acquisition is described in [X5 COCO](../../../../platforms/x5/datasets/coco/README.md) and [S COCO](../../../../platforms/s/datasets/coco/README.md); datasets are not included here.
+Use a held-out COCO-format instance dataset, with `images`, `categories` and `annotations`. Each image needs integer `id`, relative `file_name`, `width` and `height`; each category needs integer `id` and `name`. Box/mask scoring needs valid instance ground truth, including bounding boxes, segmentation, area and crowd flags. Predictions-only mode accepts an image/category manifest with an empty annotations list. Dataset acquisition is described in the unified [COCO](../../../../datasets/coco/README.md) guide, with the archived [X5 COCO](../../../../platforms/x5/datasets/coco/README.md) and [S COCO](../../../../platforms/s/datasets/coco/README.md) snapshots as provenance; datasets are not included here.
 
 PF class IDs are **not COCO category IDs**. Supply a reviewed mapping that binds the fixed [4585-class vocabulary](../test_data/classes.names), checking both source and destination names. [mapping.example.json](mapping.example.json) demonstrates person (PF 2163 → COCO 1) and chair (PF 821 → COCO 62). It covers two categories only and is **not a complete COCO-80 mapping**. Replace/extend it for your annotation categories:
 
diff --git a/samples/vision/yoloe/evaluator/README_cn.md b/samples/vision/yoloe/evaluator/README_cn.md
index 3734c8f2..04bb51ad 100644
--- a/samples/vision/yoloe/evaluator/README_cn.md
+++ b/samples/vision/yoloe/evaluator/README_cn.md
@@ -7,7 +7,7 @@
 <a id="dataset"></a>
 ## 数据集与类别映射
 
-使用独立验证集，格式为包含 `images`、`categories`、`annotations` 的 COCO 实例标注。图片需要整数 `id`、相对路径 `file_name`、`width`、`height`；类别需要整数 `id` 和 `name`。框/掩码计分需要有效的实例框、分割标注、面积和 crowd 标记。只导出预测时，可以提供 annotations 为空的图片/类别清单。数据准备参考 [X5 COCO](../../../../platforms/x5/datasets/coco/README.md) 与 [S COCO](../../../../platforms/s/datasets/coco/README.md)，数据集不随本仓库提供。
+使用独立验证集，格式为包含 `images`、`categories`、`annotations` 的 COCO 实例标注。图片需要整数 `id`、相对路径 `file_name`、`width`、`height`；类别需要整数 `id` 和 `name`。框/掩码计分需要有效的实例框、分割标注、面积和 crowd 标记。只导出预测时，可以提供 annotations 为空的图片/类别清单。数据准备参考统一的 [COCO](../../../../datasets/coco/README_cn.md) 指南，X5/S 平台快照（[X5 COCO](../../../../platforms/x5/datasets/coco/README.md)、[S COCO](../../../../platforms/s/datasets/coco/README.md)）保留作溯源；数据集不随本仓库提供。
 
 PF 类别 ID **不是 COCO category ID**。必须提供经过检查的映射，以固定 [4585 类词表](../test_data/classes.names)为依据，同时核对源/目标名称。[mapping.example.json](mapping.example.json) 演示 person（PF 2163 → COCO 1）和 chair（PF 821 → COCO 62）；它只有两个类别，**不是完整 COCO-80 映射**。请按自己的标注类别扩展或替换：
 
```

## 10. Boundaries of this evidence
- Documentation-only remediation; no dataset download, board run, toolchain, OE/HMCT or quantization validation was executed.
- Sample contract checkers pass with the pre-existing policy-skip sets unchanged; no prose test beyond the static checks in section 7.
- H8 is NOT closed by this package; independent Codex review pending.
- Report: docs/releases/unified-migration/2026-09-28-datasets-docs-remediation.md

## 11. DATASET-R1/R2/R3 remediation (2026-09-28, post independent review)

Independent review: docs/releases/unified-migration/2026-09-28-datasets-independent-review.md (changes-required). Fixes applied to README pairs only; scripts/labels/images/.gitignore untouched.

### Corrections made
- R1: dotav1 EN/CN no longer claim an official numeric category-ID 1–15 annotation order; native DOTA annotations carry category names (authority: official page supplied by the reviewer). Root EN/CN dotav1 row and class-index bullet corrected. "mislabels every category" replaced: orders coincide only at index 0 (plane), 14 of 15 positions differ. Author evidence section 4 header corrected.
- R2: coco EN/CN now state .gitignore covers only the direct val2017/annotations layout, not the script's coco_full/ output; recommend out-of-checkout working directory or a local .git/info/exclude entry. Root EN/CC intro no longer implies downloads are automatically excluded.
- R3: PascalVOC EN language switch no longer loops to itself (now points to README_cn.md). ImageNet EN/CN separate runtime display-name --label-file (dict/one-per-line, shared loader) from the Ultralytics classification evaluator's --label-file (ordered n######## synset IDs; display-name file cannot be passed) and its --val-txt ground truth; zebra described as a smoke input with per-model recorded evidence, not a guarantee.

### Re-verification
```
--- R1 counterexample (same-index names across the two DOTA files) ---
same: [(0, 'plane')] | differing positions: 14
--- R2 ignore results (this tree, unchanged .gitignore) ---
coco_full rc=1 (1 = NOT ignored, as stated)
.gitignore:33:datasets/coco/val2017/*	datasets/coco/val2017/example.jpg
val2017 rc=0 (0 = ignored)
--- R3 language navigation first lines ---
==> datasets/PascalVOC/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/PascalVOC/README_cn.md <==
[English](./README.md) | 简体中文

==> datasets/imagenet/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/imagenet/README_cn.md <==
[English](./README.md) | 简体中文

==> datasets/coco/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/coco/README_cn.md <==
[English](./README.md) | 简体中文

==> datasets/dotav1/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/dotav1/README_cn.md <==
[English](./README.md) | 简体中文

==> datasets/yoloe/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/yoloe/README_cn.md <==
[English](./README.md) | 简体中文

==> datasets/README.md <==
English | [简体中文](./README_cn.md)

==> datasets/README_cn.md <==
[English](./README.md) | 简体中文
--- stale-phrase scan across datasets/ (expect no matches) ---
no stale phrases
no stale format-equivalence claims
```

Static checks re-run after fixes: actual outputs recorded in section 12 below.

## 12. Post-remediation check outputs (2026-09-28)
```
OK: all local links and anchors resolve (16 files)
check_links rc=0
OK: anchors, heading counts and code-block counts match in all 6 pairs
check_parity rc=0
OK: executable command lines byte-identical in all pairs
check_cmds2 rc=0
git diff --check: clean
--- self-link scan (language switches must not point at own file) ---
self-loop check: OK (12 files, no self-loops)
--- reviewer-noted ignore behavior, restated ---
coco_full ignored rc=1 (expect 1)
val2017 ignored rc=0 (expect 0)
```
--- affected sample contract checkers, fresh run (sample-side text unchanged this round) ---
summary: 1 samples, 0 violations, 2 skips, 0 exemptions applied
summary: 1 samples, 0 violations, 1 skips, 0 exemptions applied
```

## 13. R2 follow-up correction (exclude-path example, 2026-09-28)

Literal '>> .git/info/exclude' replaced with cwd/worktree-independent $(git rev-parse --git-path info/exclude) form in datasets/coco/README{,_cn}.md; out-of-checkout invocation retained as primary recommendation. No exclude file, .gitignore or script modified; no download run.
```
--- .git is a file in this worktree ---
.git: ASCII text
--- git-path resolution is cwd-independent (read-only) ---
/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.git/info/exclude
/Users/Max/Workspace/company/development/RDK_MODEL_ZOO/rdk_model_zoo/.git/info/exclude
--- literal-path scan across datasets/ (expect no matches) ---
no literal .git/info/exclude appends remain
--- new form present in both languages ---
datasets/coco/README.md:1
datasets/coco/README_cn.md:1
--- static checks ---
OK: all local links and anchors resolve (16 files)
OK: anchors, heading counts and code-block counts match in all 6 pairs
OK: executable command lines byte-identical in all pairs
git diff --check: clean
```
