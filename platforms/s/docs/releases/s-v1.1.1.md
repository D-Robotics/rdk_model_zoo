# RDK Model Zoo S v1.1.1

This release corrects the S-series release metadata and manifest summary while
preserving the benchmark values and precision records from the existing release
evidence. The Ultralytics model artifact metadata is aligned with the
committed download script: 94 existing `yolov8`/`yolov9`/`yolov10` filenames and
URLs are normalized, two unsupported `yolo9n` assets are removed, and the two
published `yolov9t` S100/S100P assets are added.

## Audited inventory

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

## Validation and limitations

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
