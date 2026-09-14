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
- See [release notes](docs/releases/s-v1.1.2.md).

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
- See [release notes](docs/releases/s-v1.1.2.md).

## [s-v1.1.1] - 2026-09-07

- Corrected the S-series release metadata and manifest summary for the
  `s-v1.1.1` release candidate.
- Aligned the Ultralytics model filenames and URLs with the committed download
  script, excluded unsupported `yolo9n` assets, and added the published
  `yolov9t` S100/S100P assets.
- Audited 473 benchmark records with 1,037 performance metrics and 1,682
  accuracy metrics; existing evidence values, model assets and download URLs
  are unchanged.
- See [release notes](docs/releases/s-v1.1.1.md) for inventory totals,
  immutable evidence references, checksum coverage and validation scope.

## [s v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see s-v1.1.0.md).


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
