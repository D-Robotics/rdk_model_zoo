# Changelog

## [x3-v1.1.2] - 2026-09-09

- Record the per-thread BPU latency/throughput that the README_cn `RDK X3`
  tables publish for YOLOv8n, YOLOv10n and YOLOv8n-seg (previously only the
  summary FPS and post-processing latency were recorded).
- Rename post-processing entries to `post_process_latency` to match the
  X5/S naming, so they stop rendering as BPU latency rows.
- See [release notes](#x3-v1.1.2-details) for scope and validation.

## [x3-v1.1.1] - 2026-09-07

- Publish refreshed model inventory with corrected release metadata and summary counts.
- See [release notes](#x3-v1.1.1-details) for scope, validation and checksum coverage.

## [x3 v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see [details](#x3-v1.1.0-details)).


## x3-v1.0.0 — 2026-09-04

This release records the RDK X3 model and benchmark baseline present on the
`rdk_x3` branch at the release commit.

- Added a machine-readable model inventory at [`release/models.yaml`](docs/manifests/models.yaml).
- Added X3-only performance and accuracy evidence at [`release/benchmarks.yaml`](docs/manifests/benchmarks.yaml).
- Added the release schemas under [`release/schemas/`](docs/manifests/schemas).
- Marked this snapshot as a historical, legacy, non-normalized inventory.
- No board tests were run as part of release preparation.

## Archived version details

Full historical notes are retained below. Paths and statements describe their original version; later entries may supersede them. The v1.1.0 tags are preparation snapshots without a published GitHub Release.

<a id="x3-v1.1.2-details"></a>

<details>
<summary>x3-v1.1.2</summary>

### RDK Model Zoo X3 v1.1.2

This release completes the performance evidence that the X3 sample
documentation already publishes, and aligns the manifest's metric naming with
the X5 and S release lines. No measured value is changed, re-scaled, inferred
or removed.

#### What changed

- **Per-thread BPU measurements recorded** for `yolov8n-x3`, `yolov10n-x3`
  and `yolov8n-seg-x3`: the `README_cn` `RDK X3 & RDK X3 Module` tables
  publish latency/throughput at 1, 2, 4 and 8 threads
  (for example YOLOv8n: 99.8 ms / 10.0 FPS single-thread through
  231.0 ms / 34.1 FPS at 8 threads). The manifest previously carried only the
  summary FPS (now annotated as the 8-thread measurement it is) and the
  post-processing latency — 21 new performance entries across the three
  records.
- **Post-processing entries renamed** from `latency` (scope
  `post-processing (Python)`) to `post_process_latency`, the name the X5 and
  S manifests already use for the same column, so the online catalog renders
  them as evidence rather than as a BPU latency row. Values, units and scopes
  are unchanged.
- **Evidence paths corrected** for the YOLOv5, FCOS and the three records
  above: the per-thread tables live in each sample's `README_cn` `RDK X3`
  section (the English `README.md` republishes the RDK X5 tables), so the
  `source.path` of those records now points at `README_cn.md`. The referenced
  files are byte-identical at the recorded `source.ref`
  (`a34e5f67f07418004650f97b069fc61ceeb2bd26`), which is unchanged.

#### Audited inventory

- 15 model entries and 20 benchmark records, unchanged.
- Performance metrics 83 → 104; accuracy metrics 25, unchanged.
- Assets 20 (19 downloadable), SHA-256 coverage unchanged (0 recorded).

#### Validation and limitations

- Both manifests pass schema validation; identifiers remain unique and every
  benchmark asset reference still resolves.
- Every new value was transcribed verbatim from the `README_cn` X3 sections
  and re-verified by a deterministic table extractor that diffs each markdown
  table cell against the manifest (0 missing, 0 mismatched, 0 unmatched rows
  across all 15 X3 samples).
- The English sample READMEs label their benchmark tables `RDK X5 & RDK X5
  Module` even though they live in this repository; the manifest keeps using
  the `README_cn` X3 sections as the source of truth for X3 hardware.
- No repository-wide RDK board test was run, and none of these numbers
  certifies current runtime behavior or board compatibility.

Previous published release: [x3-v1.1.1](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/x3-v1.1.1).

</details>

<a id="x3-v1.1.1-details"></a>

<details>
<summary>x3-v1.1.1</summary>

### RDK Model Zoo X3 v1.1.1

#### Changes / 更新

- Publishes the refreshed X3 inventory prepared in v1.1.0, with corrected release dates, summary counts and immutable evidence references. The existing v1.1.0 tag is retained as a preparation snapshot; it is not retagged.
- 15 samples; 20 manifest assets (19 direct download entries); 20 benchmark records containing 83 performance and 25 accuracy measurements. These are manifest totals, not unique online model-family counts.
- Includes the refreshed YOLOv8n / YOLOv10n detection and YOLOv8n-seg bbox/mask accuracy records.

#### Validation and limitations / 验证范围

- Release preparation checks metadata, schemas, source references and website build; no repository-wide board test was run. Existing benchmark values and model download URLs are unchanged from x3-v1.1.0.
- Model SHA-256 coverage is incomplete: 0 recorded, 20 unknown. The attached SHA256SUMS checks the two YAML release attachments only, not external model binaries.
- Model files remain hosted at their existing external addresses. Consult each sample for runtime requirements.

Previous published release: [x3-v1.0.0](https://github.com/D-Robotics/rdk_model_zoo/releases/tag/x3-v1.0.0).

</details>

<a id="x3-v1.1.0-details"></a>

<details>
<summary>x3-v1.1.0</summary>

### RDK Model Zoo X3 v1.1.0

Patch data refresh for the X3 line. No board tests were run.

#### Changed

- Added missing detection accuracy to `release/benchmarks.yaml`: YOLOv8n mAP 37.3, YOLOv10n mAP 38.5, and YOLOv8n-seg bbox/mask mAP (36.7 / 30.5).

</details>

<a id="x3-v1.0.0-details"></a>

<details>
<summary>x3-v1.0.0</summary>

### RDK Model Zoo X3 v1.0.0

发布日期 / Release date：2026-09-04
分支 / Branch：`rdk_x3`
版本 / Version：`1.0.0`
标签 / Tag：`x3-v1.0.0`

#### 发布范围 / Release scope

这是 RDK X3 分支的历史基线发布。`rdk_x3` 是旧版、非标准化的示例集合；
本发布保留该版本中可追溯的模型家族、公开下载资产以及文档中的 X3 指标，
不表示所有示例当前仍可构建或得到硬件验证。

This is the historical baseline release for the RDK X3 branch. `rdk_x3` is a
legacy, non-normalized collection of demos. The release preserves the model
families, public download assets, and X3 metrics traceable in that snapshot;
it does not claim that every demo remains buildable or hardware validated.

清单包含 15 个逻辑模型家族、19 个可下载外部二进制资产，以及 1 个手动 FCOS 资产：

The manifests contain 15 logical model families, 19 downloadable external
binary assets, and one manual FCOS asset explicitly documented by the branch:

- 图像分类 / Image classification：GoogLeNet、MobileNetV1、MobileNetV2、MobileNetV4、MobileOne、RepGhost、RepVGG、RepViT、ResNet。
- 目标检测 / Object detection：FCOS、PaddleOCR、YOLOv5、YOLOv8、YOLOv10。
- 实例分割 / Instance segmentation：YOLOv8-Seg。

模型和指标清单见 [`release/models.yaml`](https://github.com/D-Robotics/rdk_model_zoo/blob/x3-v1.0.0/release/models.yaml) 与
[`release/benchmarks.yaml`](https://github.com/D-Robotics/rdk_model_zoo/blob/x3-v1.0.0/release/benchmarks.yaml)。GitHub Release
页面同时附加这两个文件，便于下载、审阅和自动化消费。

The model and metric manifests are [`release/models.yaml`](https://github.com/D-Robotics/rdk_model_zoo/blob/x3-v1.0.0/release/models.yaml)
and [`release/benchmarks.yaml`](https://github.com/D-Robotics/rdk_model_zoo/blob/x3-v1.0.0/release/benchmarks.yaml). The GitHub
Release attaches both files for download, review, and automated consumption.

#### 指标和来源 / Metrics and sources

`benchmarks.yaml` 只收录 X3 表格中的数值。每条记录的 `source.ref` 均为
`x3-v1.0.0`，`source.path` 和 `source.section` 指向本发布源代码中现存的
Markdown 文件和 ATX 标题。分类模型使用各自模型 README 的 X3 性能表；
FCOS、PaddleOCR 和 YOLOv5 使用对应模型 README 的 X3 表格；YOLOv8、
YOLOv10 和 YOLOv8-Seg 使用仓库 README 的 X3 摘要表。

`benchmarks.yaml` records only values from the X3 tables. Every record uses
`x3-v1.0.0` as `source.ref`, and its `source.path` and `source.section` point
to an existing Markdown file and ATX heading in this release source. The
classification entries use the X3 performance table in each model README;
FCOS, PaddleOCR, and YOLOv5 use their model README tables; YOLOv8, YOLOv10,
and YOLOv8-Seg use the X3 summary tables in the repository README.

未在 X3 文档中给出数值的精度或性能保持为空。分类精度使用文档中给出的
浮点和量化 Top-1；YOLOv5 的 X3 表格给出的浮点 mAP（50-95）也被记录。
文档没有为所有记录提供完整的运行时条件，因此缺失条件不会由清单推断。

Accuracy or performance values absent from the X3 documentation remain empty.
Classification accuracy uses the documented floating-point and quantized
Top-1 values; the floating-point mAP (50-95) values in the YOLOv5 X3 table
are also recorded. Runtime conditions are incomplete for some documentation,
so missing conditions are not inferred by the manifests.

#### 资产限制 / Asset limitations

清单只保留分支中明确记录的 URL；没有从上游项目扩展变体，也没有补写
校验和。FCOS 的 `fcos_512x512_nv12.bin` 只有本地复制命令，因此标记为
手动资产。下载脚本和模型下载文档的路径均按发布源代码保存。ACT 与 Pi0
在该 X3 快照中没有可纳入的本地模型证据。

The manifests retain only URLs explicitly recorded by the branch. They do not
expand upstream variants or add checksums. `fcos_512x512_nv12.bin` has only a
local copy command, so it is marked as a manual asset. Download script paths
and model documentation paths are preserved from the release source. ACT and
Pi0 have no local model evidence eligible for this X3 snapshot.

#### 验证范围 / Validation scope

本发布只进行了清单结构、引用、来源文件和标题、平台隔离及 Git 空白检查。
没有运行 RDK 板卡测试；X3 结果是已有文档的历史记录。

Release preparation checked manifest structure, references, source files and
headings, platform isolation, and Git whitespace. No RDK board tests were run;
the X3 results are historical records transcribed from existing documentation.

</details>
