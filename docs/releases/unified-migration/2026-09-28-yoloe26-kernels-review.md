# YOLOE-26 PF internal kernel extraction

Base: `48bbbb1df63307d3759e300b3f2d3a4918b19972`. This is an implementation
self-check, not independent acceptance or a finished canonical YOLOE migration.
User scope remains: retire duplicate standalone detection/segmentation/pose
implementations, preserve YOLOE, YOLO-World and YOLO26 Depth.

## Behavior and documentation

The new shared geometry and numerical modules separate fixed 4585-class PF
preprocessing, Top-K candidate decoding and ROI mask reconstruction from model
loading. They preserve round/114 letterboxing, direct LTRB, no NMS, single- or
multiple-label selection and source tie ordering. Masks interpolate logits before
zero thresholding, crop in model coordinates and restore binary pixels with
nearest interpolation. These rules differ from ordinary YOLO26 segmentation.

Per-call immutable geometry records concrete resize dimensions. Box inverse
scaling uses actual per-axis resize ratios instead of the source ideal common
gain. Source-equal non-rounding cases and explicit rounded expected values are
tested separately. Outputs own their memory; invalid shapes, integer tensors,
nonfinite values and malformed options fail explicitly. Logical shape metadata
requires integral dimensions, rejecting bool and float equality lookalikes.

The shared README includes matching English/Chinese protocol, input/output,
reproduction and limitation sections. There is no sample entrypoint yet. Current
S published quantized HBM outputs are not accepted by these floating-only
kernels; no manual dequantization path was added and no new float asset was
built. Preserve the float-output conversion and validation gap from the
[artifact audit](2026-09-28-yoloe-source-review.md).

## Validation

Nine new host checks compare pinned S source candidate arrays, preprocessing
pixels, calibration tensors and nontrivial mask pixels; they also exercise
ownership, invalid input and actual resize geometry. The source runtime SHA-256
is checked before loading its pure functions. No source snapshot is modified.

Final regression: 417 tests passed (Ultralytics 141, shared 153, ResNet 52,
OCR 44, checker 27). Migration checker: 44 samples, zero violations, 46
explicit skips, zero exemptions. The existing Ultralytics bilingual README
link/example harness also passed; this does not validate unimplemented YOLOE
customer guides. Complete commands, UTC timestamps, exit codes and logs are in
[evidence](evidence/2026-09-28-yoloe26-kernels/host-results.json);
[file hashes](evidence/2026-09-28-yoloe26-kernels/result.json) bind the code.
The initial missing-module red test is retained. A later metadata boundary test
exposed Python bool/float tuple equality; integral validation fixed it before
the final regression.

## Remaining work

Canonical X5/S YOLOE stages, artifact binding, conversion, CLI/evaluator, native
capabilities and complete customer README layers remain pending. Board, actual
SDK/OE, dataset accuracy and independent review remain not-run. B9/H5/H8 and
H0–H9 stay open; foundational numerical tests do not close a sample migration.
