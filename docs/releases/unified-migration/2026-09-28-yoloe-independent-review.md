# YOLOE independent host runtime review

Reviewer: Codex. Baseline e5ff50f2. Disposition: accept the reviewed host runtime
composition and entrypoint scope. H5/B9 overall, conversion/evaluator document
review and full repository integration remain open. No product edits by reviewer.

## Verification

- 34 Python tests passed across runtime, pinned-source behavior comparison,
  bilingual runtime examples, native launcher and preflight modules. The sample
  checker reports 0 violations, 1 CLI policy skip and 0 exemptions. All 95 YOLOE
  sample file hashes stayed stable during this run.
- Rebuilt the existing native stage-library configuration with real cached OpenCV,
  ASan/UBSan and warnings-as-errors. All 11 CTests passed: float heads, E11/E26
  decoding, geometry, preflight, image operations, pipeline, both SDK doubles,
  CLI IO and CLI help. Linker duplicate-library warnings are preserved in logs.
- [Python evidence](evidence/2026-09-28-yoloe-independent-review/runtime-tests.json)
  and [native evidence](evidence/2026-09-28-yoloe-independent-review/native-tests.json)
  record commands, complete outputs and current candidate hashes. Native product
  and shared SDK-fixture paths have no working diff after the run.

## Inspected contracts

The Python task holds only construction and pre_process/forward/post_process/
predict. IO, configuration, binding and decoding have separate modules. Predict
carries the per-image transform explicitly; forward performs one raw call. Native
Prepared and RawBatch carry a private shared task identity and reject cross-task
use; SDK output is read through the shared output binding before ownership moves
into the result. SDK setup checks preflight, a single named model, the target's
640-square packed/split NV12 input, and ten exact float output heads.

YOLOE-11 keeps DFL16 and classwise NMS; X5 full-image masks and S ROI masks remain
separate source behaviors. YOLOE-26 uses direct offsets and Top-K without NMS;
configuration rejects unsupported NMS, morphology and resize choices. The
4585-class prompt-free vocabulary is not advertised as arbitrary text prompting.
The source comparison and SDK fixtures exercise these distinctions; they cannot
prove hardware execution or accuracy.

README parameters distinguish library/CLI morphology defaults, original-image
coordinates, raw borrowed Python outputs, X5 packed transport, S planes and target
variants. Both runtime library examples execute under an injected SDK. Published
S models remain quantized and are explicitly rejected by the floating route;
a separately identified float conversion is required. The existing artifact gap
is retained rather than casting integer heads or treating a host fixture as
published-model validation. Real conversion execution is excluded by the user's
latest scope and is not a completion requirement for this review.

## Remaining scope

This is runtime composition/entrypoint acceptance, not acceptance of every
conversion/evaluator instruction or the complete B9 migration. Continue the
source-depth documentation audit and shared integration checks. No weights were
downloaded; no board/robot, real SDK, calibration, compiler or quantization
accuracy run occurred. Historical author results retain their own original scope.
