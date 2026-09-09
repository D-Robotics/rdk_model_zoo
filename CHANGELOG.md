# Changelog

## [x3-v1.1.2] - 2026-09-09

- Record the per-thread BPU latency/throughput that the README_cn `RDK X3`
  tables publish for YOLOv8n, YOLOv10n and YOLOv8n-seg (previously only the
  summary FPS and post-processing latency were recorded).
- Rename post-processing entries to `post_process_latency` to match the
  X5/S naming, so they stop rendering as BPU latency rows.
- See [release notes](docs/releases/x3-v1.1.2.md) for scope and validation.

## [x3-v1.1.1] - 2026-09-07

- Publish refreshed model inventory with corrected release metadata and summary counts.
- See [release notes](docs/releases/x3-v1.1.1.md) for scope, validation and checksum coverage.

## [x3 v1.1.0] - 2026-09-07

- Data refresh: benchmark/accuracy values re-transcribed from sample READMEs (see x3-v1.1.0.md).


## x3-v1.0.0 — 2026-09-04

This release records the RDK X3 model and benchmark baseline present on the
`rdk_x3` branch at the release commit.

- Added a machine-readable model inventory at [`release/models.yaml`](release/models.yaml).
- Added X3-only performance and accuracy evidence at [`release/benchmarks.yaml`](release/benchmarks.yaml).
- Added the release schemas under [`release/schemas/`](release/schemas/).
- Marked this snapshot as a historical, legacy, non-normalized inventory.
- No board tests were run as part of release preparation.
