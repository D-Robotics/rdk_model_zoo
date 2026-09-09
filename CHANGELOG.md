# Changelog

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
- See [release notes](docs/releases/x5-v1.1.2.md).

## [x5-v1.1.1] - 2026-09-07

- Publish refreshed model inventory with corrected release metadata and summary counts.
- See [release notes](docs/releases/x5-v1.1.1.md) for scope, validation and checksum coverage.

## [x5 v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see x5-v1.1.0.md).


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
