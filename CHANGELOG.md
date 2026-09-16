# Changelog

## [x5-v1.1.3] - 2026-09-16

- Release the refreshed dashboard, platform/family statistics, configuration-based Benchmark tables and representative model details from rdk_x5.
- Include source-backed catalog corrections and fresh-checkout CI fixes; retain S/X3 v1.1.2 pins.
- See [release notes](#x5-v1.1.3-details) for inventory and validation scope.

## [x5-v1.1.2] - 2026-09-08

- Catalog detail tables now show every measurement a model publishes. Accuracy
  values whose stage the source does not label (`model_stage` absent) were being
  dropped from the table and reported as "not yet measured" while still visible
  in the row evidence; 69 such measurements across SigLIP, DiffusionDrive,
  EfficientSAM, MobileSAM, PointNet, Paraformer, KWS and YOLO26 are now shown.
- Accuracy columns are derived from each model's own published metrics instead
  of one shared three-column template, so a classifier shows Top-1/Top-5, a
  detector shows the bbox all/small/medium/large columns of its source table, a
  pose model shows the keypoint columns and an embedding model shows cosine
  similarity. One measurement per cell; a measurement is kept out of a row whose
  timing scope does not cover it.
- Accuracy values render exactly as published. They were previously re-scaled
  into percentages (`0.391` shown as `39.1 %`); the scale now appears once in the
  column header and only retention keeps a `%`.
- Official model names: YOLO11/12/13/26 drop the `v` that YOLOv5/8/9/10 keep,
  ResNet18/50/152 are capitalised, `PaddleOCR v6` becomes `PaddleOCR`, and the
  X3/X5 `paddleocr` and S `paddle_ocr` slugs merge into one card. Family ids are
  unchanged so published deep links keep working.
- Row names no longer repeat the selected hardware (`YOLOv8n Detect on RDK S100`
  under the S100 tab).
- S-series catalog input advanced to `s-v1.1.2`, which records the retention
  percentage the S source tables already publish. X3 stays at `x3-v1.1.1`.
- X5 manifest values are unchanged in this release.
- See [release notes](#x5-v1.1.2-details).

## [x5-v1.1.1] - 2026-09-07

- Publish refreshed model inventory with corrected release metadata and summary counts.
- See [release notes](#x5-v1.1.1-details) for scope, validation and checksum coverage.

## [x5 v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see [details](#x5-v1.1.0-details)).


All notable changes to each platform release are recorded in this file. RDK Model Zoo follows [Semantic Versioning](https://semver.org/) independently for each maintained platform branch.

## [X5 1.0.0] - 2026-09-03

### Added

- Established the first formal versioned snapshot of the `rdk_x5` delivery branch.
- Published 35 vision samples and one robotics sample for RDK X5.
- Added an X5 model artifact manifest with download locations and all SHA256 values already maintained by the repository.
- Added the release policy, tag convention, release checklist, and withdrawal procedure.

### Compatibility

- Target hardware: RDK X5.
- Recommended system: RDK OS 3.5.0 or later, based on Ubuntu 22.04 aarch64 and TROS-Humble.
- Runtime and OpenExplorer requirements remain sample-specific and are documented in each sample directory.

### Validation scope

- This release records the current maintained source and published model inventory.
- No repository-wide board test was performed for this release. Individual sample documentation remains the source for sample-specific validation results.

[X5 1.0.0]: https://github.com/D-Robotics/rdk_model_zoo/releases/tag/x5-v1.0.0

## Archived version details

Full historical notes are retained below. Paths and statements describe their original version; later entries may supersede them. The v1.1.0 tags are preparation snapshots without a published GitHub Release.

<a id="x5-v1.1.3-details"></a>

<details>
<summary>x5-v1.1.3</summary>

### RDK Model Zoo X5 v1.1.3

This patch releases the refreshed model dashboard from `rdk_x5`. The public entry remains https://d-robotics.github.io/rdk_model_zoo/ . The main-branch restructuring and documentation-site migration are not included.

- Home overview shows model-family and configuration counts, including platform breakdowns.
- Model details consolidate BPU and CPU post-processing measurements into configuration rows, preserve separate timing columns, and move repeated timing conditions below the table.
- YOLOv8, Paraformer, HIMLoco and SigLIP receive task-appropriate detail layouts; ACT platform coverage and Pi0 annotations are corrected.
- Source-backed catalog corrections restore omitted measurements and correct model names and asset references. Published measurements retain provenance; no new board performance is claimed.
- Fresh-checkout CI builds the catalog before running data-dependent UI tests and creates its own ignored audit output directory.

The combined dashboard contains **54 families, 584 configurations and 820 benchmark records**. It pins X5 to `x5-v1.1.3`, S to `s-v1.1.2`, and X3 to `x3-v1.1.2`; existing tags remain unchanged.

The X5 manifest contains 37 samples, 177 assets (176 downloadable, one manually supplied), and 239 benchmark records. Trusted SHA256 values are recorded for 24 assets; 153 remain without repository-recorded checksums.

Validation uses the dashboard test suite, TypeScript checks, production build and release-identity check. This is a catalog/UI release; sample runtime code and the develop-branch YOLO consolidation are not part of it. No repository-wide board inference or accuracy validation is claimed.

Release attachments: `models.yaml` and `benchmarks.yaml`.

</details>

<a id="x5-v1.1.2-details"></a>

<details>
<summary>x5-v1.1.2</summary>

### RDK Model Zoo X5 v1.1.2

This release fixes how the online catalog presents measurements that were
already published. No X5 manifest value changes; the X5 inventory totals are
identical to `x5-v1.1.1`. The S-series catalog input advances to `s-v1.1.2`
(completing the post-processing and SigLIP evidence it publishes) and X3
advances to `x3-v1.1.2` (recording the per-thread BPU measurements its
README_cn X3 tables publish).

Combined catalog release tag: `x5-v1.1.2+s-v1.1.2+x3-v1.1.2`.

#### Changes / 更新

##### Published data was reported as missing

`pairAccuracyMetrics` accepted only `model_stage: float` and `quantized`, so an
accuracy value whose stage the source does not label was dropped from the table
and shown as "not yet measured" — while the same value remained visible in the
row's evidence expansion. 69 measurements were affected across SigLIP,
DiffusionDrive, EfficientSAM, MobileSAM, PointNet, Paraformer, KWS and YOLO26.
Pairs now carry an `other` bucket, and a value with no comparable counterpart
reports "not applicable" rather than implying it was never measured.

##### One template for every model

Accuracy columns are now derived from the measurements a model actually
publishes. A classifier shows Top-1/Top-5, a detector shows the
`bbox all/small/medium/large mAP@.50:.95` columns of its source table, a pose
model shows the keypoint columns and an embedding model shows cosine similarity.
Each measurement gets its own cell instead of several metrics sharing one, and a
measurement is kept out of a row whose timing scope does not cover it — SigLIP
reports Top-1/Top-5 for `pooler output` and cosine similarity for
`last hidden state`, and those no longer appear on each other's rows.

##### Values are shown as published

Accuracy values were re-scaled into percentages, so a source value of `0.391`
was displayed as `39.1 %`, and the raw evidence table showed `74.82 %` next to
`unit: ratio`. Values now render exactly as recorded, with dataset and scale
stated once in the column header. Retention keeps its `%` because it is defined
as a percentage. A missing FPS on a row whose latency was measured now reads
"not recorded" instead of "performance not yet measured".

##### Official model names

Names were synthesised from directory slugs. They now follow the spellings the
sample READMEs publish: `YOLO11`, `YOLO12`, `YOLO13` and `YOLO26` drop the `v`
that `YOLOv5`/`YOLOv8`/`YOLOv9`/`YOLOv10` keep; `Resnet18/50/152` become
`ResNet18/50/152`; `PaddleOCR v6` becomes `PaddleOCR`. Row names no longer
repeat the selected hardware (`YOLOv8n Detect on RDK S100` under the S100 tab).
Family ids are unchanged, so published deep links such as `?model=yolov26` keep
working. The X3/X5 `paddleocr` and S `paddle_ocr` sample slugs now merge into
one card instead of publishing two cards with the same name.

##### One row per measurement, not fragments

Latency and FPS of one measurement were split across several rows whenever the
manifests worded their timing scope differently (`multi-thread` vs no scope,
`BPU; 2 threads` vs `BPU task`, `frame rate`, `100 frames`). A GoogLeNet X3
row rendered as three rows — 8.34 ms, 16.29 ms and 243.51 FPS each on their own
line. Scope wording that only restates the thread configuration now folds into
one row (the thread columns stay distinct), an unstated statistic merges with
the stated mean, and a second latency that cannot share a thread bucket keeps
its own row instead of being hidden. Genuinely different contexts — encoder
vs decoder, pooler vs last hidden state, compiler estimates, min/p50/p95/max
statistics — stay separate rows.

##### Cross-platform row consistency

Row names are normalised to one spelling per model: X5's `YOLOv8n-CLS`,
S's `Cls` and the asset-derived `Classify` all render `YOLOv8n CLS`;
`Resnet18`/`MobileNetv4 medium` become `ResNet18`/`MobileNetV4 Medium`;
S100P SigLIP rows carry the same `patch16/patch14` wording as S100; and
`PP-OCRv3_det` becomes `PP-OCRv3 Detection`. The X5 manifest's two
RDK-X3-hardware paddleocr records no longer duplicate the X3 rows (each
platform's manifest is authoritative for its own hardware only).

##### S-series evidence completed by s-v1.1.2

The S accuracy tables print a retention percentage beside every `FP32 / BPU`
pair (`0.309 / 0.291 (94.3 %)`), but the manifest stored only the two values, so
the catalog derived an approximation (94.17 % for YOLOv8n on S100 instead of the
published 94.3 %). `s-v1.1.2` records the published value for all 768 S
float/quantized pairs and adds 12 `bbox-small/medium/large` values for YOLOv8n
and YOLO11n on S100 that the source publishes. See
[s-v1.1.2](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/s-v1.1.2).

#### Audited inventory

- X5: 36 samples, 174 manifest assets (173 direct download entries), 227
  benchmark records with 623 performance and 401 accuracy measurements —
  unchanged from `x5-v1.1.1`.
- Combined catalog: 53 model family cards across X5, S100, S100P, S600 and X3
  after the PaddleOCR merge.
- S `s-v1.1.2` adds 194 `post_process_latency` values (the CPU Latency
  column), the SigLIP PyTorch TOP1/TOP5 baseline and MSE column, and unified
  SigLIP scopes.
- X3 `x3-v1.1.2` records the per-thread BPU latency/throughput of YOLOv8n,
  YOLOv10n and YOLOv8n-seg and renames post-processing entries to
  `post_process_latency`.
- Model SHA-256 coverage is still incomplete on X5: 11 recorded, 163 unknown.

#### Validation and limitations

- `npm run check` passes: 136 tests across 20 files, TypeScript strict
  typecheck, and a production Vite build. Regression tests lock the behaviours
  above (including ten from the sample-by-sample self-review pass), and the
  naming rules are asserted against the production catalog rather than a
  fixture.
- Every markdown benchmark table cell of all 86 samples across the three
  release lines was diffed deterministically against the manifests after the
  changes: 0 missing, 0 mismatched, 0 unmatched rows (the gemma4-e2b
  conversion-tutorial accuracy values stay unrecorded by policy — a tutorial
  is not a benchmark publication and states no board).
- Release preparation validates metadata, schemas, source refs and the website
  build. No repository-wide board test was run.
- The catalog reads X5 from this release checkout and pins S/X3 to the immutable
  annotated tags in `docs/release/catalog-sources.json`.
- Where a source publishes no value, the catalog still says so. X5 detection
  tables do not publish a small/medium/large breakdown, so X5 rows show
  `bbox-all` only; that is a property of the source, not a gap in this release.

Previous published release: [x5-v1.1.1](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/x5-v1.1.1).

</details>

<a id="x5-v1.1.1-details"></a>

<details>
<summary>x5-v1.1.1</summary>

### RDK Model Zoo X5 v1.1.1

#### Changes / 更新

- Publishes the refreshed X5 inventory prepared in v1.1.0, with corrected release dates, summary counts and immutable evidence references. The existing v1.1.0 tag is retained as a preparation snapshot; it is not retagged.
- 36 samples; 174 manifest assets (173 direct download entries); 227 benchmark records containing 623 performance and 401 accuracy measurements. These are manifest totals, not unique online model-family counts.
- Restores documented top-1 accuracy for HGNetV2, RepGhost and RepVGG. UNet accuracy remains omitted where usable units/environment were not established; no accuracy values are invented.
- Locks S/X3 catalog inputs to versioned tags, reads X5 from the release checkout, and fixes aggregate-version verification during Pages deployment.

#### Validation and limitations / 验证范围

- Release preparation checks metadata, schemas, source references and website build; no repository-wide board test was run. Existing benchmark values and model download URLs are unchanged from x5-v1.1.0.
- Model SHA-256 coverage is incomplete: 11 recorded, 163 unknown. The attached SHA256SUMS checks the two YAML release attachments only, not external model binaries.
- Model files remain hosted at their existing external addresses. Consult each sample for runtime requirements.

Previous published release: [x5-v1.0.0](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/x5-v1.0.0).

</details>

<a id="x5-v1.1.0-details"></a>

<details>
<summary>x5-v1.1.0</summary>

### RDK Model Zoo X5 v1.1.0

Patch data refresh for the multi-platform online catalog. No board tests were run; every value is transcribed from the tagged sample READMEs.

#### Changed

- Multi-platform catalog (`docs/catalog`) now reads S and X3 benchmark data from their live release branches (`rdk_s`, `rdk_x3`) instead of the stale `s-v1.0.0` / `x3-v1.0.0` tags, matching the existing X5 live-branch read.
- Restored missing top-1 accuracy in `docs/release/benchmarks.yaml` for HGNetV2, RepGhost, RepVGG (16 records) and added UNet ResNet18 accuracy plus ResNet34/50/101/152 records.

#### Known limitations

- Repository-wide board validation is not claimed; consult each sample README.
- Model binaries are hosted outside Git and most download scripts still do not record a SHA256 digest (`sha256: null`).

</details>

<a id="x5-v1.0.0-details"></a>

<details>
<summary>x5-v1.0.0</summary>

### RDK Model Zoo X5 v1.0.0

This is the first formal versioned snapshot of the `rdk_x5` delivery branch for RDK X5. The release provides model conversion guides, model download scripts, and Python/C++ runtime examples for common vision and robotics workloads.

#### Included

- 35 vision samples covering image classification, object detection, segmentation, pose estimation, depth estimation, OCR, image matting, and image-text matching.
- The HIMLoco Unitree Go2 locomotion policy sample under `samples/robotics/himloco`.
- A versioned [model manifest](https://github.com/D-Robotics/rdk_model_zoo/blob/x5-v1.0.0/release/models.yaml) that maps every sample included in this release to its model provisioning instructions and artifacts. The same file is attached to this Release as `models.yaml`.
- A [benchmark manifest](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/release/benchmarks.yaml) that transcribes performance and accuracy values already published in the tagged sample documentation, preserving missing conditions instead of inferring them.
- The public [online model catalog](https://d-robotics.github.io/rdk_model_zoo/) with searchable cards, model assets, benchmark conditions, and immutable links back to this Release Tag.
- SHA256 digests for the artifacts whose download scripts already maintain verified checksums. Artifacts without a trusted repository digest are marked `sha256: null`.

#### Compatibility

- Target hardware: RDK X5.
- Recommended system: RDK OS 3.5.0 or later, based on Ubuntu 22.04 aarch64 and TROS-Humble.
- Runtime requirements are sample-specific and normally use the BSP-compatible `hbm_runtime` or `libdnn`.
- Model conversion uses the RDK X5 OpenExplorer toolchain version documented by each sample.
- Deployment artifacts are generally X5-specific `.bin` models; a few samples also use ONNX models.

#### Getting started

Clone or check out this tag, choose a sample, and follow its README. Downloadable model binaries are obtained separately from the D-Robotics archive by scripts in each sample's `model` directory. MODNet is marked `availability: manual` in the manifest and must be provided manually.

```bash
git clone --branch x5-v1.0.0 --depth 1 https://github.com/D-Robotics/rdk_model_zoo.git
cd rdk_model_zoo/samples/vision/ultralytics_yolo/model
bash download_model.sh
```

#### Known limitations

- This release does not claim repository-wide board validation. Consult each sample README for its documented validation status.
- Model binaries are hosted outside Git. Most existing download scripts do not yet record a SHA256 digest; the manifest marks those values as `null`.
- Download URLs are sample-specific and are not versioned by the Git tag.
- Many samples use fixed input shapes, INT8 quantization, and NV12 input. Results can differ from floating-point, dynamic-shape, or RGB baselines.
- Operators unsupported by the BPU can fall back to CPU. Sample benchmark figures may exclude preprocessing and postprocessing.

---

### RDK Model Zoo X5 v1.0.0（中文）

这是面向 RDK X5 的 `rdk_x5` 主交付分支首个正式版本化快照，包含模型转换说明、模型下载脚本，以及常见视觉与机器人任务的 Python/C++ 运行示例。

#### 本版本内容

- 35 个视觉样例，覆盖图像分类、目标检测、分割、姿态估计、深度估计、OCR、图像抠图和图文匹配。
- `samples/robotics/himloco` 下的 HIMLoco Unitree Go2 运动控制策略样例。
- 版本化的[模型清单](https://github.com/D-Robotics/rdk_model_zoo/blob/x5-v1.0.0/release/models.yaml)，将本版本纳入的每个样例映射到模型获取说明和模型文件；同一文件也作为 `models.yaml` 附件随 Release 发布。
- [Benchmark 清单](https://github.com/D-Robotics/rdk_model_zoo/blob/rdk_x5/release/benchmarks.yaml)整理本 Tag 样例文档已经公开的性能与精度值；来源未说明的条件保持缺失，不进行推算。
- 公开的[在线模型目录](https://d-robotics.github.io/rdk_model_zoo/)，提供模型卡片搜索、模型资产、Benchmark 条件和指向本 Release Tag 的不可变来源链接。
- 收录当前下载脚本中已经维护的 SHA256；没有可信仓库校验值的文件明确标记为 `sha256: null`。

#### 兼容性

- 目标硬件：RDK X5。
- 推荐系统：RDK OS 3.5.0 或更高版本，基于 Ubuntu 22.04 aarch64 和 TROS-Humble。
- 运行时依赖以各样例文档为准，通常使用与 BSP 匹配的 `hbm_runtime` 或 `libdnn`。
- 模型转换使用各样例文档指定的 RDK X5 OpenExplorer 工具链。
- 部署产物通常是 X5 专用 `.bin` 模型，少量样例还会使用 ONNX 模型。

#### 开始使用

克隆或切换到本 Tag，选择样例并按照对应 README 操作。可下载的模型二进制文件独立托管，由各样例 `model` 目录中的脚本从 D-Robotics 文件服务器获取。MODNet 在清单中标记为 `availability: manual`，需要手工放置模型文件。

```bash
git clone --branch x5-v1.0.0 --depth 1 https://github.com/D-Robotics/rdk_model_zoo.git
cd rdk_model_zoo/samples/vision/ultralytics_yolo/model
bash download_model.sh
```

#### 已知限制

- 本版本不声明完成全仓库范围的板卡验证；各样例的验证状态以对应 README 为准。
- 模型二进制文件托管在 Git 之外。多数现有下载脚本尚未记录 SHA256，清单中将其标记为 `null`。
- 下载地址由各样例分别维护，尚未由 Git Tag 对模型文件本身进行版本化。
- 许多样例采用固定输入尺寸、INT8 量化和 NV12 输入，结果可能与浮点、动态尺寸或 RGB 基线不同。
- BPU 不支持的算子可能回退到 CPU；样例 Benchmark 可能不包含前处理和后处理。

</details>

<a id="v1.1.1-release-assessment-details"></a>

<details>
<summary>v1.1.1-release-assessment</summary>

### 三平台 v1.1.1 发版评估

#### 版本决策

三个 `v1.1.0` 附注 Tag 已推送，但尚未创建 GitHub Release。保留这些准备快照；发布元数据和流程修正使用独立的 `x5-v1.1.1`、`s-v1.1.1`、`x3-v1.1.1`，不移动旧 Tag。

#### 本轮需完成

- 同步 VERSION、两份清单的发布身份、日期、实际统计和 CHANGELOG。
- 修正发布说明与最新内容的矛盾，例如 X5 的 UNet 精度实际未收录。
- 将 benchmark 证据固定到原始完整提交 SHA，保持测量值不变。
- 将网页已有依据的 S 模型文件名修正落实到发布清单，使 benchmark 关联与附件一致。
- X5 从自身检出构建，S/X3 通过 catalog-sources.json 锁定附注 Tag；禁止活动分支漂移。
- Pages 验证组合目录中的 X5 平台版本，而非用组合标签与单个 Tag 比较。
- 发布两份 Tag 原始 YAML 和 SHA256SUMS；校验远端附件、Tag、分支和在线目录。

#### 后续工作

- 补齐外部模型文件的 SHA-256，并在实际文件变更时保存可追溯版本；本轮附件哈希只覆盖 YAML。
- 按原始证据整理 ACT/Pi0 的任务分类，目前清单将其归为足式运动控制。
- 模型列表继续更新后显式推进对应硬件版本，再更新在线目录的版本锁定；不要把新的分支内容继续挂在旧版本名下。
- 板卡测试与新增精度评测作为独立工作，不列入这次简化发布的门禁，也不宣称已经完成。

</details>
