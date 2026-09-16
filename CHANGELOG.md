# Changelog

## [s-v1.1.2] - 2026-09-08

- Recorded the retention percentage that the source accuracy tables already
  publish beside every `FP32 / BPU` pair. 768 published retention values were
  added across 180 benchmark records.
- Added the 12 accuracy values the source publishes but the manifest missed:
  `bbox-small/medium/large mAP@.50:.95` for YOLOv8n and YOLO11n on S100.
- Added the 65 benchmark records the two Ultralytics evaluator READMEs publish
  but the manifest never carried: the whole Classification task for
  `ultralytics_yolo` on S100/S100P (previously S600-only) and for
  `ultralytics_yolo26` on S100/S600, plus the full S100P detection accuracy
  table for `ultralytics_yolo26`. 136 performance and 200 accuracy metrics,
  transcribed verbatim.
- Audit: one verification agent per sample across all 35 S samples proved
  every remaining record matches its source row; the remaining 97 documented
  gaps (SigLIP float/MSE columns, per-case and Total-Latency columns,
  code-fenced cosines) are listed in the release note for the next batch.
- Benchmarks 473 -> 538 records; performance 1,037 -> 1,173; accuracy
  1,682 -> 2,668 (834 retention). No published value was changed or inferred.
- See [release notes](#s-v1.1.2-details).

- Recorded the retention percentage that the source accuracy tables already
  publish beside every `FP32 / BPU` pair. 768 published retention values were
  added across 180 benchmark records; until now the catalog could only derive
  an approximation from the two accuracy values.
- Added the 12 accuracy values the source publishes but the manifest missed:
  `bbox-small/medium/large mAP@.50:.95` for YOLOv8n and YOLO11n on S100
  (`ultralytics_yolo/evaluator/README.md`, detection accuracy table).
- Accuracy metrics 1,682 -> 2,468 over the same 473 benchmark records and
  1,037 performance metrics. No published value was changed, re-scaled or
  removed, and no measurement was inferred.
- Retention was matched by evidence, not by name: a source row is used only
  when its device matches the record hardware, its model label appears in the
  record's display name, and the row reproduces the record's own published
  float/quantized numbers exactly. 0 of 768 pairs were ambiguous after that
  check; YOLOv13 uses its own `Pytorch | YUV420SP Python (retention)` layout
  and one table covering both S100 and S100P.
- See [release notes](#s-v1.1.2-details).

## [s-v1.1.1] - 2026-09-07

- Corrected the S-series release metadata and manifest summary for the
  `s-v1.1.1` release candidate.
- Aligned the Ultralytics model filenames and URLs with the committed download
  script, excluded unsupported `yolo9n` assets, and added the published
  `yolov9t` S100/S100P assets.
- Audited 473 benchmark records with 1,037 performance metrics and 1,682
  accuracy metrics; existing evidence values, model assets and download URLs
  are unchanged.
- See [release notes](#s-v1.1.1-details) for inventory totals,
  immutable evidence references, checksum coverage and validation scope.

## [s v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see [details](#s-v1.1.0-details)).


All notable changes to the RDK Model Zoo S-series release line are recorded in
this file.

## [1.0.0] - 2026-09-04

### English

- Published the first formal S-series baseline from the immutable `s-v1.0.0`
  source ref on `rdk_s`.
- Added a repository-evidence model inventory covering 33 releasable samples,
  308 published or locally provisioned assets, and the matching benchmark
  manifest.
- Recorded checksum availability explicitly; unknown SHA-256 values are `null`.
- This release validates manifests and documentation only. It does not include
  a repository-wide RDK board test gate.

### 简体中文

- 基于 `rdk_s` 分支的不可变 `s-v1.0.0` 源引用，发布首个正式的 S 系列基线版本。
- 新增基于仓库证据的模型清单，覆盖 33 个可发布 sample、308 个已发布或本地提供的资产，以及对应的 benchmark 清单。
- 显式记录校验和覆盖情况；未知 SHA-256 值统一写为 `null`。
- 本版本只校验清单与文档，不设置全仓库 RDK 板卡测试门禁。

[1.0.0]: https://github.com/D-Robotics/rdk_model_zoo/releases/tag/s-v1.0.0

## Archived version details

Full historical notes are retained below. Paths and statements describe their original version; later entries may supersede them. The v1.1.0 tags are preparation snapshots without a published GitHub Release.

<a id="s-v1.1.2-details"></a>

<details>
<summary>s-v1.1.2</summary>

### RDK Model Zoo S v1.1.2

This release completes the accuracy evidence that the S-series source tables
already publish. Every `FP32 / BPU` cell in the Ultralytics and YOLOv13
accuracy tables prints its retention percentage beside the two values
(`0.309 / 0.291 (94.3 %)`); the manifest stored only the two values, so the
online catalog had to derive an approximation instead of showing the published
number. This release records the published one.

No measured value is changed, re-scaled, inferred or removed. The release adds
evidence that was already in the repository documentation.

#### What changed

- **768 published retention values added** across 180 benchmark records, one for
  each float/quantized accuracy pair the manifest already carried. These are
  recorded as `<metric>-retention` with `unit: percent`, matching the existing
  X5 convention.
- **65 missed benchmark records added**: the entire Classification task for
  `ultralytics_yolo` on S100 and S100P (previously carried for S600 only) and
  for `ultralytics_yolo26` on S100 and S600, plus the complete S100P
  detection accuracy table for `ultralytics_yolo26` — 136 performance and
  200 accuracy metrics found by the per-sample audit agents and transcribed
  verbatim from the evaluator READMEs.
- **194 post-processing latencies added**: every Ultralytics and YOLOv13
  performance row publishes a `CPU Latency (Single Core)` column
  (2.0 ms detection, 5.0 ms segmentation, 1.0 ms pose, 0.5 ms classification;
  YOLOv13 2.0 ms) that the manifest did not carry. These are recorded as
  `post_process_latency` with scope `single-core CPU`, matching the X5
  convention, for all 186 Ultralytics and 8 YOLOv13 performance records.
- **SigLIP accuracy completed**: the PyTorch baseline TOP1/TOP5 column
  (16 values, `model_stage: float`) and the `last hidden state` MSE column
  (8 values, mean) that the accuracy tables publish beside the already
  recorded BPU and cosine values — 48 entries across the 16 records
  (each variant carries both the S100 and S100P record).
- **12 missed accuracy values added**: `bbox-small`, `bbox-medium` and
  `bbox-large` `mAP@.50:.95` (FP32 and BPU) for YOLOv8n and YOLO11n on S100,
  plus their 6 published retentions. The source detection table publishes all
  four bbox columns for every row; these two rows had only `bbox-all` recorded.
- Accuracy metrics 1,682 → 2,468 (of which 774 are retention) from the
  retention work, → 2,668 with the 65 new Ultralytics records, and → 2,716
  with the SigLIP float/MSE completion. Performance metrics 1,037 → 1,173
  with the new records, → 1,367 with the post-processing column. Benchmark
  record count (538) and model entries (35) are unchanged.

#### How each value was matched

Retention was matched by evidence, never by name similarity or row order:

1. A source table row is a candidate only when its device column equals the
   record's hardware, its model label appears in the record's `display_name`,
   and the row reproduces at least one of the record's own published
   float/quantized pairs exactly.
2. A model's measurements are spread over several tables that share one label
   (`##### bbox`, `##### mask`, pose, classification), so all rows matching one
   label contribute columns to that record.
3. A retention is attached only to the column whose float and quantized numbers
   are exactly the numbers already stored for that metric.

After this, 768 of 768 pairs resolved to exactly one published retention, with
0 ambiguous and 0 unresolved. Value-only matching is not sufficient on its own:
6 `(device, float, quantized)` triples occur in more than one table with
retentions differing in the last decimal (for example S100 `0.461 / 0.441` is
published as both 95.5 % and 95.6 %), and the label check is what separates
them.

YOLOv13 uses a different layout — float in the `Pytorch` column and
`quantized (retention)` in the `YUV420SP Python` column — and its single table
is headed `### RDK S100 / RDK S100P`, so it applies to both boards. Those
retentions are recorded to two decimals as published (`93.27`).

#### Audited inventory

- 35 model entries: 33 with download scripts and 2 manual entries (`act` and
  `pi0`).
- 308 manifest assets: 306 with download URLs and 2 local-only assets
  (`s100/am.mvn`, `s100/paraformer_config.yaml`).
- SHA-256 coverage is still incomplete: 2 recorded, 306 `null`.
- 538 benchmark records contain 1,367 performance metrics and 2,716 accuracy
  metrics (of which 834 are retention).

Benchmark evidence remains anchored to the full commit SHA
`53d924f4c88175ec77634d24e5d711a3a0901eb6`. The three source documents read for
this release (`ultralytics_yolo`, `ultralytics_yolo26` and `yolov13_imoonlab`
evaluator READMEs) are byte-identical at that commit and at `s-v1.1.1`, so the
existing refs stay valid and were not rewritten.

#### Validation and limitations

- Both manifests pass schema validation; model and benchmark identifiers remain
  unique and every benchmark asset reference still resolves.
- The change is 786 added lines and no deletions or reformatted lines, so each
  added value can be read against its source row.
- Retention values published to one decimal are stored as published. The catalog
  previously showed a derived value (for example 94.17 % for YOLOv8n on S100);
  it now shows the published 94.3 %.
- No repository-wide RDK board test was run, and none of these numbers certifies
  current runtime behavior or board compatibility.
- The release `SHA256SUMS` attachment covers the two YAML manifest attachments
  only, not external model binaries.

Previous published release: [s-v1.1.1](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/s-v1.1.1).

#### Known remaining gaps (audited, not yet recorded)

The per-sample audit (one verification agent per sample, all 35 S samples)
confirmed every value now in the manifest matches its source row, and
documented 97 further published numbers not yet recorded. This release
records the SigLIP column (24) and the YOLOv13 CPU latency, and the audit's
second deterministic pass found the Ultralytics/YOLOv13 post-processing
columns (194 entries) recorded here as well. What still remains:

- SigLIP: the PyTorch float-stage TOP1/TOP5 column and the MSE column (24
  values; the manifest carries the BPU values and cosine).
- Paraformer: C++ UCP stage latencies, RTF values and the FP32 baseline CER
  (14). DiffusionDrive: per-case S600 BEV metrics (8). DINOv2: six
  board-executed cosine values and the PTQ calibrated cosine (8).
- ResNet152 evaluator runtime table (7), gemma4-e2b S100P performance and
  quantization accuracy (6), yolo26_depth board latencies (5), Total-Latency
  columns for 3dresnet/depth_anything_v2/pointnet (9), calibrated cosines for
  mobilenetv2/v3/v4 (4), vit float ONNX accuracy (2), yolov13 CPU latency
  column (2).

Each carries file, heading and verbatim cell evidence in the audit reports.

</details>

<a id="s-v1.1.1-details"></a>

<details>
<summary>s-v1.1.1</summary>

### RDK Model Zoo S v1.1.1

This release corrects the S-series release metadata and manifest summary while
preserving the benchmark values and precision records from the existing release
evidence. The Ultralytics model artifact metadata is aligned with the
committed download script: 94 existing `yolov8`/`yolov9`/`yolov10` filenames and
URLs are normalized, two unsupported `yolo9n` assets are removed, and the two
published `yolov9t` S100/S100P assets are added.

#### Audited inventory

- 35 model entries: 33 with download scripts and 2 manual entries (`act` and
  `pi0`).
- 308 manifest assets: 306 with download URLs and 2 local-only assets. The
  local-only assets are `s100/am.mvn` and `s100/paraformer_config.yaml` from
  Paraformer.
- SHA-256 coverage is incomplete: 2 assets have recorded SHA-256 values and
  306 remain `null` because no trusted digest is recorded in the repository.
- 473 benchmark records contain 1,037 performance metrics and 1,682 accuracy
  metrics. The previous summary values (58, 188 and 97) were stale.

Benchmark evidence is anchored to the full commit SHA
`53d924f4c88175ec77634d24e5d711a3a0901eb6`, which is the commit resolved by
the immutable `s-v1.1.0` tag. This keeps the original evidence source stable
while the release metadata advances to `s-v1.1.1`.

#### Validation and limitations

- Model and benchmark YAML files pass schema validation, model and benchmark
  identifiers are unique, and all benchmark asset references resolve after the
  source-backed Ultralytics metadata correction above. The 70 YOLO26 records
  now point to the existing `## Benchmark Results` heading. The DINOv2 subheading was also checked against the original document;
  no measured value is changed.
- All benchmark source paths exist. The source refs use the immutable commit
  recorded above, so they remain resolvable independently of the new release
  tag.
- No repository-wide RDK board tests were run. The benchmark values document
  existing repository evidence and do not certify current runtime behavior or
  board compatibility.
- The release `SHA256SUMS` attachment covers the two YAML manifest attachments
  only; it does not provide checksums for external model binaries.
- The GitHub Release attaches `models.yaml` and `benchmarks.yaml` so the exact
  release inventory can be downloaded with these notes.

</details>

<a id="s-v1.1.0-details"></a>

<details>
<summary>s-v1.1.0</summary>

### RDK Model Zoo S v1.1.0

Benchmark data refresh for the S100 / S100P / S600 line. No board tests were run; values are transcribed from the `rdk_s` sample READMEs and evaluator documents.

#### Changed

- `docs/release/benchmarks.yaml` expanded from 58 to 473 records: Ultralytics YOLO (368), YOLO26 (90), SigLIP (16), YOLO26 Depth (20), YOLOv13 (8), plus per-device S100/S100P/S600 rows.
- `docs/release/models.yaml` adds ACT and Pi0 (rdk_LeRobot_tools submodules).
- Removed erroneous `cls-640` classification rows (copy-paste of the 640x640/80-class detection config).

#### Known limitations

- Several S samples are runtime demos without a published board benchmark; those remain unmeasured in this release.

</details>

<a id="s-v1.0.0-details"></a>

<details>
<summary>s-v1.0.0</summary>

### RDK Model Zoo S v1.0.0

#### English

This is the first formal S-series baseline release from the `rdk_s` branch.
It freezes source ref `s-v1.0.0` on 2026-09-04 and records the repository
inventory in `release/models.yaml` and `release/benchmarks.yaml`.

Tagged manifests: [models.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/models.yaml) and [benchmarks.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/benchmarks.yaml).

The release contains 33 releasable sample entries and 308 model or supporting
assets: 306 downloadable references and 2 files provisioned locally by a
committed helper. Unknown SHA-256 values are written as `null` and are not
treated as verified checksums. ACT and Pi0 remain external gitlink samples and
are omitted from these totals. No source files were changed to represent these
external samples.

The benchmark manifest includes numeric performance and accuracy evidence that
already appears in repository documentation. It preserves the original test
conditions where they are stated and leaves unpublished or non-numeric values
unreported. This release has no repository-wide RDK board test gate and makes
no board validation claim.

The GitHub Release attaches both `models.yaml` and `benchmarks.yaml` so that
the exact release inventory can be downloaded alongside these notes.

#### 简体中文

这是 `rdk_s` 分支的首个正式 S 系列基线版本。版本固定于 2026-09-04，源引用为
`s-v1.0.0`，模型清单与 benchmark 清单分别位于 `release/models.yaml` 和
`release/benchmarks.yaml`。

带标签的清单链接：[models.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/models.yaml) 和 [benchmarks.yaml](https://github.com/D-Robotics/rdk_model_zoo/blob/s-v1.0.0/release/benchmarks.yaml)。

本版本包含 33 个可发布 sample 和 308 个模型或配套资产，其中 306 个为可下载引用，
2 个由仓库内提交的脚本在本地提供。未知 SHA-256 值统一写为 `null`，不表示已完成校验。
ACT 和 Pi0 是外部 gitlink sample，不计入上述总数；为表示这些外部 sample，没有修改源代码。

benchmark 清单只收录仓库文档中已经公开的数值化性能与精度证据，并保留原文已经说明的测试条件；
未公开或无法数值化的内容不作推断。本版本不设置全仓库 RDK 板卡测试门禁，也不作板卡验证声明。

GitHub Release 会同时附加 `models.yaml` 与 `benchmarks.yaml`，便于下载与本版本对应的完整清单。

</details>
