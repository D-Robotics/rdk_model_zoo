# RDK Model Zoo S v1.1.2

This release completes the accuracy evidence that the S-series source tables
already publish. Every `FP32 / BPU` cell in the Ultralytics and YOLOv13
accuracy tables prints its retention percentage beside the two values
(`0.309 / 0.291 (94.3 %)`); the manifest stored only the two values, so the
online catalog had to derive an approximation instead of showing the published
number. This release records the published one.

No measured value is changed, re-scaled, inferred or removed. The release adds
evidence that was already in the repository documentation.

## What changed

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

## How each value was matched

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

## Audited inventory

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

## Validation and limitations

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

## Known remaining gaps (audited, not yet recorded)

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
