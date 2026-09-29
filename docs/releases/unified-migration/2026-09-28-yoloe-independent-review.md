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

## Evaluator and subdirectory follow-up (2026-09-28)

Codex read the model, conversion and evaluator instructions in both languages and
compared public options/defaults/status/output descriptions with the preparation
parser and evaluator parser, backend, dataset, execution and scoring modules.
All 14 local README files have no missing local file links. Source performance
figures retain source conditions and failed/absent measurements; host prediction
counts are not described as dataset AP or reproduced quantized-model counts.
No actual export, calibration or compiler invocation was performed in this review.

The synthetic evaluator suite initially failed with five missing-pycocotools
errors in the default environment. The documented scorer dependency already
existed in the coordination cache; rerunning with that explicit PYTHONPATH
passed all 10 tests without installation or product changes. The initial output
is preserved in [evaluator-docs.json](evidence/2026-09-28-yoloe-independent-review/evaluator-docs.json);
[recheck](evidence/2026-09-28-yoloe-independent-review/evaluator-recheck.json)
records the environment and successful full output. These tests use synthetic
masks and a fake predictor, not a real model. Pycocotools/NumPy deprecation
warnings are retained. Accept this bounded scorer/data-contract host check.

### YOLOE-DOC-R1 — stale C++ implementation status

The conversion guides' known-gaps paragraph still says C++ migration remains
unfinished; the evaluator closing paragraph repeats that status. This contradicts
the implemented native library/entrypoint and the scoped host acceptance above.
Claude Code + GLM has been assigned a documentation-only update: distinguish
implemented and host-reviewed code from real SDK/board not-run and whole-branch
review still open. Do not remove genuine float-S-publication or hardware gaps,
change trusted conversion recipes, or claim B9 completion. Documentation closure
remains pending this specific correction and final recheck.

## YOLOE-DOC-R1 closure (2026-09-28)

Closed after reading the six revised bilingual subdirectory guides and matching
the claims against the scoped acceptance above. See the [independent status-doc
review](2026-09-28-review-status-docs-independent-review.md) for final hashes,
command invariance, link checks and fresh contract-check output. This does not
close the complete B9 migration.
