# PointNet independent host review — host scope accepted

Reviewer: Codex. Base `ba05780f`. B8/PointNet remains open. This review found a
blocking defect in the documented default CLI path; the author suite's 21 tests
passed on the current Python 3.14.7 host but do not exercise plotting.

## POINTNET-R1 — default plotting fails on supported Python versions (P1)

`runtime/python/visualization.py:13` retains an unrelated `parse_args` function
from the source CLI and annotates its return as `argparse.Namespace`, without
importing argparse. Python 3.13 raises NameError while importing the module.
`runtime/python/main.py:65` imports this module whenever `--no-plot` is absent,
which is the default quick-start path. The CLI does not catch NameError and
cannot write its final result report after this failure. README declares Python
3.10+ support. Python 3.14's deferred annotation behavior masks this import error;
passing the current host suite is therefore insufficient.

The source visualization module was imported with explicit NumPy/matplotlib
import doubles to isolate Python name resolution without installing plotting
dependencies. Actual Python 3.13.15 returned 1 with NameError; Python 3.14.7
imported successfully. This is not a plot-rendering test or board inference.
Exact fixture, source hashes, interpreter versions and full outputs are in
[evidence](evidence/2026-09-28-pointnet-independent-review/findings.json).

## Required Claude implementation and acceptance

Remove the unused copied CLI parser from the visualization helper, preserving
plot helpers and main.py as the sole CLI owner. Do not merely add argparse to
keep a second unused parser and contradictory defaults. Add meaningful coverage
of the default plotting path using injected runtime and an appropriate host
plotting backend/fixture, plus import coverage that catches supported-version
annotation failures. Verify requested original/segmented output and result report;
keep --no-plot behavior. No board or model run is required. Do not install a
quantization toolchain or change source recipe instructions.

The inspected task class already keeps pre_process, forward, post_process and
predict separate, with file IO in main and SDK access in its runner. Existing
normalization/parity and target/metadata tests pass. These observations do not
close the full sample or B8: the default-path defect must be fixed and independently
rechecked. Source diagram/benchmark preservation was documented in the author
record; whole-branch acceptance remains separate.

## Independent recheck — POINTNET-R1 closed (2026-09-28)

The original finding and failure evidence above remain historical. Codex reviewed
both changed product files and reran verification after the Claude session ended.
The unused parser was removed; plotting functions and main.py are unchanged.
The 23-test PointNet suite passes, the original Python 3.13 import reproducer now
returns 0, and the sample checker reports 0 violations, 1 CLI policy skip and
0 exemptions. Exact commands, complete output and accepted source hashes are in
[recheck.json](evidence/2026-09-28-pointnet-independent-review/recheck.json).

The new default-path test uses real NumPy and the real visualization module with
an injected SDK and recording matplotlib fixture; only the output directory is
overridden (`main(['--output-dir', tmp])`). The separate import reproducer doubles
NumPy and matplotlib. These are host import/call-level checks, not PNG rendering,
model execution or board evidence. Python 3.10–3.12 execution remains not-run.

Corrections to author narration: the shared manifest failure was independently
recorded as MiniCPM CORE-R2, not merely an assumed half-written file; it was fixed
before this live-tree recheck. MiniCPM itself still awaits separate acceptance.
The author's Claude log also records a local automatic-memory edit outside this
repository; that edit is not included or accepted as part of this package.
Subsequent worker instructions explicitly prohibit automatic-memory access.

Disposition: accept this bounded POINTNET-R1 remediation and its regression
coverage. This does not close all B8/H4 samples or whole-branch acceptance.
Board and quantization verification remain not-run under current user scope.

## POINTNET-R2 — accepted int32 logits lose ordering before argmax (P2)

Full-sample follow-up found that binding accepts int32 SCALE output, but the task
uses apply_output_transform whose dequant path casts raw values to float32 before
argmax. With accepted scalar scale=1 and zero_point=0, raw scores
[16777216,16777217,0,0] should select class 1. Production post_process selects
class 0 because float32 rounds the first two values to an artificial tie.
[Reproducer evidence](evidence/2026-09-28-pointnet-independent-review/int32-argmax.json).
This is a synthetic host tensor case, not an observed board artifact failure or
a quantization recipe check. The current claimed int32 contract makes it relevant.

Preserve integer ordering during affine decoding for argmax (the existing shared
dequantize_tensor supports float64 comparison). Keep raw inference unchanged and
F32 vestigial-descriptor behavior intact; do not change shared defaults for other
samples or silently drop the advertised integer contract. Cover int32 distinction,
per-channel scale/offset and true ties. Clarify comparison precision in the two
runtime guides, run the bounded PointNet suite and checker, and return for review.
POINTNET-R1 remains closed; full PointNet/B8 acceptance stays changes-required.

## Independent recheck — POINTNET-R2 closed (2026-09-28)

Codex inspected the four changed PointNet files after the Claude Code + GLM
process exited, then reran all 26 PointNet tests and the sample checker. The
checker reports 0 violations, 1 documented CLI policy skip, 0 exemptions.
The original accepted int32 metadata and raw [16777216,16777217,0,0] reproducer
now selects class 1; raw input remains unchanged. All 26 sample file hashes stayed
stable across verification. Full commands, outputs and hashes are in
[r2-recheck.json](evidence/2026-09-28-pointnet-independent-review/r2-recheck.json).

The integer branch now requests float64 from the existing dequantizer before
argmax. New tests cover positive/negative large integer distinctions, per-channel
scale/offset and exact decoded ties. F32 still bypasses quantization descriptors;
the shared default, binding contract, visualization and target gate were not
changed. Different per-channel transforms may of course change raw-value ranking:
the intended comparison is between decoded scores, not between raw integers.
These tests establish the reported precision regression is fixed; they do not
claim an exact-arithmetic proof for every possible quantization descriptor.

Disposition: close POINTNET-R2 and accept the reviewed PointNet host runtime and
associated documentation scope, including the previously closed R1. This is not
whole-batch B8/H4 or whole-branch closure. Python 3.10–3.12, board inference and
real quantization execution remain not-run; R1's Python 3.13 import-only evidence
retains its original limits. Original failures and author evidence are preserved.
