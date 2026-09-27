# Shared native NV12 allocation and task lifetime corrections

Base: `d0a68bd69583c5eac5e7340304f5de023da1f8cc`. This work prepares the
common SDK boundary for YOLOE and fixes existing Ultralytics callers. It does
not complete YOLOE's SDK backend/CLI, the other migrations or H0–H9 acceptance.
No board or real SDK build/run was performed; API doubles prove host control
flow and memory handling only.

## Verified defects and corrections

The unchanged baseline returned immediately when task creation failed, even
if the SDK had returned a task handle. The shared synchronous wrapper now
releases every returned handle after creation, submission or wait failure,
preserves the first error and rejects success without a handle. UCP explicitly
selects the BPU-any backend. Invalid null pointers/counts fail before SDK calls.

Split input allocation previously derived dynamic capacity from align(width),
while uploads followed the descriptor's actual row pitch. A 6-byte row with
128-byte pitch and four rows received 256 bytes instead of at least 512. Both
that capacity defect and task release omission were reproduced against the
pre-change source saved in evidence. The new common metadata resolver is used
by probe and allocation, validates geometry/dtype/quantization/inner strides,
and derives dynamic capacity from the actual pitch. The tests write all rows
under ASan/UBSan, including row padding, partial allocation and failure cleanup.
Packed X5 inputs support compact RGB-shaped NCHW/NHWC descriptors; padded packed
storage is explicitly rejected rather than guessed. Unknown/invalid metadata
is not converted into a fallback input contract.

A new exact-length `upload_planes` uploads owned Y/UV from YOLOE without an
NV12-to-I420 round trip. Existing I420 callers delegate through this path. Each
allocation stores its plan; mismatched plans and uploads without successful
allocation fail. All metadata is checked before allocating any input, and a
failed second allocation immediately frees the first.

## Evidence and limits

[Evidence directory](evidence/2026-09-28-yolo-native-io/) contains unchanged
baseline source, initial compile failure (missing new API/unused old parameter),
original defect reproductions, final production-source tests, commands and
hashes. The first new fixture run failed because its rejection helper expected
zero lifetime allocations after an already completed packed-input test; the
fixture now checks acquired/freed balance. That initial harness failure remains
in `initial-run-x5.log`; it is not reported as a product defect.

Both SDK branches compile against narrow test doubles, not vendor headers or
libraries. Existing native helper suites and Python migration regression run
alongside the new cases. Bilingual native README documents the input, failure
and test contracts. No new performance or hardware compatibility claim is made.

The first full Python regression had one static-test failure: it required the
old I420 helper names to appear in `dnn_io.cc`. With direct Y/UV upload that
spelling is no longer the contract. The test now compiles and executes actual
input/task production code against both SDK doubles, while retaining the four
callers' shared-owner routing check. The original failed log and run record are
retained; the separate final Ultralytics run confirms resolution.

Final host result: 12 native CTest cases passed; 463 Python tests passed
(29 YOLOE, 143 Ultralytics, 153 shared, 52 ResNet, 44 OCR, 27 checker,
5 export and 10 evaluator). Migration checker: 45 samples, zero violations,
47 declared policy skips, zero exemptions. README link/API checks passed.
After strengthening the I420 U/V interleaving assertion, both instrumented
input/task cases were rebuilt and rerun; Python/other native code was unchanged.
