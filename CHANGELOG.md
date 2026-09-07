# Changelog

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
