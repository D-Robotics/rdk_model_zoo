# Canonical YOLOE Python integration — implementation self-check

Base: `e75b996846c402c14a83f1f1c160544586cc9a96`. Scope is canonical Python
integration and its root/model/runtime/test-data documentation. B9 migration,
conversion, native capabilities and independent acceptance remain open.

## Implementation and preserved source behavior

The canonical `samples/vision/yoloe` now contains an exact manifest selector,
explicit download command, common raw runner adapter, three-stage task, separate
numerical/geometry helpers and separate visualization/I/O. `predict` composes the
same public pre/forward/post stages, carrying immutable per-image geometry.
Forward validates physical NV12 inputs and makes one SDK call. Returned raw
arrays borrow SDK storage; task results own their data.

Fourteen original publications are mapped explicitly: X5 11s/m/l, S100 11s and
26n/s/m/l/x, S100P 26n/s/m/l/x. S600 has no supported publication. Defaults preserve
11s on X5/S100 and choose the source 26n route on S100P; an explicit asset ID
selects its own variant when variant is absent. Conflicting variants, targets
and external model paths without identity fail.

YOLOE-11 keeps DFL16/classwise NMS, but retains X5 full-image boolean mask
probability cropping/interpolation versus S ROI binary mask semantics. S11's
source library defaults morphology off while its CLI enables it unless
`--no-morph` is passed; these two defaults remain distinct. YOLOE-26 keeps
round/114 geometry, direct LTRB, single/multiple-label deterministic Top-K without
NMS and logits-first mask interpolation. Inverse geometry uses actual integer
resize ratios, an intentional correction of source ideal-scale rounding.

The common Ultralytics binder gains one opt-in for the X5 source's accepted
NHWC RGB-shaped input descriptor. Only the YOLOE-11 contract enables it; existing
YOLO contracts keep their original NCHW policy. Physical transport remains packed
NV12. Every YOLOE output must be NHWC float32; logical roles bind by unique full
shape rather than enumeration order. Semantic mappings supplied directly to
postprocessing receive the same validation as SDK outputs.

## Float artifacts and hardware boundary

Original S publications remain quantized and fail before SDK loading through the
normal canonical entry. They may be downloaded explicitly for source comparison;
the downloader explains why that does not make them canonical float artifacts.
A separately converted float model requires its own explicit model path and
SHA-256. Its source asset ID is provenance only, not the converted file's identity.
Loading checks local hardware identity, local bytes and actual model metadata.
A hash does not certify quantization accuracy or board compatibility.

No new float S artifact was built, and no board/SDK/OE run was performed. Injected
SDK loaders exist only for host tests and are explicitly supplied by tests. They
do not claim successful hardware inference.

## README quality and host checks

Eight bilingual README files cover root, model, Python runtime and test data.
They include the 14 exact publication IDs/hashes, target/variant availability,
explicit preparation, real parser defaults, visible-result commands, stage I/O,
raw-array lifetime, mask layout, full library examples and actual failures.
Historical X5/S26 Runtime-only numbers retain source context and are not presented
as current Python end-to-end measurements. Missing canonical conversion/evaluator/
C++ content is stated explicitly with links to preserved source instructions.
It remains required migration work, not an accepted pointer-only final state.

Host tests include exact source X5 and S11 postprocessing comparisons (including
both S morphology modes), X5 preprocessing pixels, source NHWC metadata acceptance,
rejection before loading on quantized/public S and bad custom hashes, malformed
physical/semantic tensors, per-image context, borrowed raw arrays and owned final
masks, stable empty results, CLI matrix/defaults, vocabulary/rendering and both
complete README integration snippets. Reference source plus utility files are
SHA-256 pinned with full original commits in `tests/source-facts.json`.

Final commands, logs and return codes are recorded in
[evidence](evidence/2026-09-28-yoloe-python/host-results.json). The migration table
now exposes YOLOE as Refactor=in-progress, so it participates in the normal checker
scope. No exemption was introduced. The final summary/file hashes accompany the
logs in the same evidence directory. Final results: 436 Python host tests
(YOLOE 19, Ultralytics 141, shared 153, ResNet 52, OCR 44, checker 27), 45-sample
checker with zero violations / 47 declared skips / zero exemptions, and 121
publisher tests with successful source validation, typecheck, build and reproducible
catalog `catalog-v1.0.0-a40359c348416a41` (57 families / 812 benchmarks).
The existing Ultralytics README harness and the two new YOLOE library examples
both passed. Publisher warnings retain historical accuracy rows without published
dataset identities (X5 125, S 35, X3 18); no dataset facts were invented to hide them.

During implementation, tests exposed missing semantic-dictionary validation,
S11 CLI morphology drift, missing input validation before SDK execution and the
source-accepted NHWC descriptor mismatch; all were corrected before final checks.
An additional extreme-threshold source comparison exposed missing X5 confidence
clamping to [1e-6,1-1e-6]; that source behavior and bilingual explanation were restored.
The first checker pass also found bad relative source links, which were corrected.
The first full host run then found the X5 active manifest still pointing to the
old download_model.sh name. Active X5 and S11/S26 sample/download paths now point
to the canonical sample while original IDs, filenames, URLs and hashes remain
unchanged. Stale S notes about absent hashes and unmigrated entry paths were
corrected. The failed initial shared-suite log and status record are retained.
No transient failure is used as evidence of completion.

## Outstanding work

Canonical conversion/export/calibration, float-output compile configuration and
verification, evaluator workflows, S native C++ migration, full customer README
layers and fresh whole-branch independent review remain. Actual board/SDK/OE and
dataset checks are not-run. B9/H5/H8 and H0–H9 remain open; this report does not
close YOLOE migration or the user goal.
