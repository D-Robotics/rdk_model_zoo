# YOLOE native E11 candidate decoding — implementation self-check

Base: `ce93fc7fddd1713cd361b028f0a2fa6ceb959739`. This increment adds E11
candidate math and a shared owning candidate type. Native masks, SDK lifecycle,
identity gates and the application remain pending; B9/H5/H0–H9 remain open.
This is not independent review or native-runtime acceptance.

## Implementation and source boundaries

`e11_decode.h` checks ten compact finite float NHWC vectors before access,
requires 4585 classes/DFL64/coefficients32/prototype32 geometry, then reuses
Ultralytics' stabilized DFL16 expectation, box coordinate and sigmoid primitives.
Classwise NMS retains owning candidates, so mask coefficients cannot drift
from retained boxes. `candidate.h` now holds the result type shared with E26;
E26 math is otherwise unchanged. No manual dequantization or SDK access is added.

Two native-source boundaries are deliberately preserved: score >= threshold
accepts a candidate and IoU > NMS threshold suppresses a same-class candidate.
The current Python NMS suppresses IoU equality as well; no exact-boundary
cross-language equivalence is claimed. The source unordered class map and
OpenMP dynamic merge had unspecified class/tie ordering. New native output is
ascending class, descending score and original scale/anchor order for exact
ties. It is deterministic, but is not a promise to reproduce source thread
ordering or NumPy's equal-score sorting. These differences are documented in
both READMEs and covered by boundary fixtures.

## Verification

`test_e11_decode.cc` was first compiled without the implementation header and
failed with a missing-header diagnostic; after implementation it passed under
AddressSanitizer/UndefinedBehaviorSanitizer. It checks uniform/extreme DFL,
same-class suppression versus different-class preservation, coefficient alignment,
inclusive score threshold, native IoU equality, deterministic ties, invalid
thresholds/caps (E26 regression), short vectors and nonfinite outputs.

The real ONNX comparison uses E11s/m/l exports on the bundled office image.
C++ and Python candidate counts are 38/26/25; labels and order agree exactly.
Box maximum absolute errors are respectively 9.1552734375e-5, 6.103515625e-5 and
9.1552734375e-5; all satisfy rtol=1e-5, atol=1e-4. Score errors are at most
2.9802322387695312e-8 (rtol=atol=1e-6), and mask coefficients are identical in
this check. Native and Python candidate records are persisted, together with
model/image/raw-output hashes, full commands, stdout/stderr, timestamps and
implementation hashes. Raw tensors and weights remain in ignored local storage.
The E26n real-output single/multi-label comparison is repeated after moving the
shared candidate type; its original criteria are unchanged.

All three sanitizer-instrumented native tests and the 29 existing YOLOE tests
passed. Migration checks reported 45 samples, zero violations, 47 declared skips
and zero exemptions; all 102 YOLOE local links passed. E26 single/multi-label
regression retained 213/300 candidates, exact class/order agreement and zero
observed box/score/coefficient difference on its one-image comparison.

The evidence runner extracts both bilingual README command blocks, checks they
match, compiles/runs all three native tests with sanitizers, rebuilds the probe,
executes both real-output comparisons, and runs the existing YOLOE tests,
migration contract checker and README link/API checks. Complete records are in
[evidence](evidence/2026-09-28-yoloe-cpp-e11/).

## Remaining scope

The native README now documents E11 and E26 separately, their parameter and
boundary differences, ownership and host commands. This candidate check does
not generate masks, restore geometry, execute the BPU, build against the real
SDK, establish dataset AP or prove original quantized-model equivalence. E11m/l
host output checks do not publish new S native artifacts or enable new targets.
All remaining migration, README and independent-review work continues.
