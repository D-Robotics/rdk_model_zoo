# B8 UNet / PP-LiteSeg independent host review — accepted within host scope

Reviewer: Codex. Reviewed committed sample implementations at `6c2bdceb`;
concurrent MiniCPM/Gemma/Agent-entry work is outside this review. No sample product
file was edited by the reviewer. B8 as a whole and the completion plan remain open.

## Scope and disposition

Accept the current UNet and PP-LiteSeg host migration scope. Inspection covered
selection/binding, separated task/runner/CLI/visualization responsibilities,
canonical evaluator reuse, model preparation documentation and the bilingual
root/model/runtime/conversion/evaluator guides. No blocking finding was identified
in this scope. This is not board execution, a measured accuracy result or a claim
that real SDK inference was exercised on this host.

UNet preserves the five X5 ResNet-backbone selections, packed NV12 stretch geometry
and fixed 512×512 VOC mask. Its task exposes the three stages and predict without
file IO; per-call frozen context is separate from state. Raw float32 logits are not
dequantized twice; validated integer SCALE outputs are transformed in postprocess,
then class argmax chooses the first class on a tie. NCHW/NHWC output layouts are
bound explicitly. The evaluator reuses the canonical X5 task while separately
identifying caller-provided artifacts instead of claiming publisher authentication.
The README matches that contract, including original checkpoint/benchmark scope,
model-resolution output and the custom-filename backbone argument.

PP-LiteSeg preserves a different boundary: packed NV12 input and already-decoded
int32 class IDs shaped (1,512,1024,1). The task rejects logits/invalid IDs and does
not apply argmax, softmax or dequantization. Visualization retains the source's
three-panel view outside inference. The evaluator is a compatibility wrapper
around the canonical entry, not an invented dataset mIoU implementation. Its
model guide correctly distinguishes an unknown publisher digest from an observed
local digest; runtime/evaluator guides explain the exact output/report scope.

## Independent evidence

[Evidence directory](evidence/2026-09-28-b8-segmentation-independent-review/) holds
complete command output, candidate source hashes and local-link inventory:

- UNet: 20 tests pass. These include source preprocessing/postprocessing parity,
  independent packed NV12 bytes, layouts and explicit-stage/predict equality,
  integer-transform placement, per-call context, strict target/metadata checks,
  SDK-free CLI and executable README API examples with a declared SDK fixture.
- PP-LiteSeg: 18 tests pass, including source tensor/mask and visualization parity,
  stage semantics, binding/CLI and README integration coverage.
- Each sample contract check: zero violations, one documented CLI policy skip,
  zero exemptions. Heading checks support but do not replace this content review.
- Twenty-two bilingual README files: 70 local Markdown link destinations resolve.
  Anchor semantics and remote URL availability are not asserted by that scan.
  The earlier bilingual command audit and source illustration inventory provide
  additional scoped evidence, not independent claims of runtime execution.

The tests use delivered inputs, host arrays and runtime doubles. No model was
downloaded, no board was contacted and no export/OE/compiler/quantized-accuracy
recipe was executed. Ordinary host preparation/argument tests are not conversion
recipe validation. Existing recipes remain trusted source documentation under the
user's current ruling; their lack of a new real conversion run is not a blocker.

## Remaining boundaries

Board remains not-run under the current environment. Whole-B8 independent review
still needs the other sample families; H0–H9 whole-branch acceptance is separate.
No new C++ or S support is claimed for these X5-only Python samples. Historical
precision/performance and source recipe gaps retain their original conditions.
The shared ledger/plan is concurrently owned by another implementation package;
roll this scoped disposition into the next coordinated ledger update rather than
overwriting that worker's in-progress file.
