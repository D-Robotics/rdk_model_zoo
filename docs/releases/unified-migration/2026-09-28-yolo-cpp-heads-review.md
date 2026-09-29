# C++ pose/segment output binding and ownership

Implementation self-check, 2026-09-28; base `929a2aa84a3c3456803d80ebc26b8ed08cb09d92`.
H2 and H0–H9 remain open; this is not independent whole-branch approval.

## Findings and implementation

Both programs previously assumed output order `[cls,box,extra]` at each scale,
read outputs as compact floats without validating type/quantization/strides,
and ignored allocation/cache errors. Segment additionally assumed prototype index
9. Early exits leaked output/model resources; image saving could report success
without checking the returned boolean.

`common/task_outputs.h` now owns pure shape-role binding and bounded finite-float
copying; `task_output_binding.h` adapts X5 aligned shape/S byte strides and owns
acquired output allocations. Nine pose or ten segment heads are bound by exact
NHWC geometry/channels, with one consistent DFL or LTRB box protocol across all
scales. Bind before allocation; failed binding cannot authorize allocate/read.
Counts, batch, geometry, dtype, quantization, positive strides, row/cell overlap,
allocation span and finite valid values are checked. Physical padding is skipped.
Copied outputs own their compact buffers; prototypes remain NHWC-only here.

Both mains use that binding and resource owner plus the packed-model owner from
the preceding increment. Output/query/cache errors and failed image saving exit
nonzero. Duplicate DFL softmax code is removed; both use the existing shared box
helpers. Keypoint math, NMS, border rejection and rendering behavior are retained.
Modified C++ functions and new helper/test code are formatted with clang-format.

## Customer documentation

Both C++ READMEs describe required shapes, unsupported NCHW prototypes, temporary
output-copy memory, error/lifetime behavior and the difference from Python.
Segment renders a model-input-sized three-panel image and uses class-agnostic NMS;
it does not provide original-image ROI masks. Both references reject boxes crossing
the input boundary. Pose retains its original resize/padding inverse arithmetic.
The prior general assertion that all tasks restore original-image coordinates
was corrected. This transport change is not a claim of full Python equivalence.

## Verification

- [Native records](evidence/2026-09-28-yolo-cpp-heads/native-results.json): ten
  C++11 test executables, all compiled and executed under ASan/UBSan (20 zero
  exits). Six pure tests plus four descriptor/resource tests using narrow X5/UCP
  doubles. They do not use real board SDK headers/libraries.
- New tests cover reordered pose/segment heads, direct/DFL variants, mixed and
  duplicate/missing roles, incompatible geometry, padded reads with NaN padding,
  owned results, undersized/overlapping strides, nonfinite data, partial allocation
  failure, descriptor/cache errors, cleanup and rejected post-failure allocation.
- Shared DFL math is compared with the former pose/segment normalization sequence
  for 200 deterministic four-side fixtures, tolerance `1e-5` in model-bin units.
- [Host records](evidence/2026-09-28-yolo-cpp-heads/host-results.json): 408 tests
  (Ultralytics 141, shared 144, ResNet 52, OCR 44, checker 27), migration checker
  44 samples / 0 violations / 46 policy skips / 0 exemptions, 128 README links and
  16 bilingual executable Python examples passed. After C++ formatting the six
  static C++ contract tests were additionally rerun successfully; native tests
  were rerun after the final binding-state guard change.

Real SDK/full executable compilation, OpenCV rendering execution, board inference
and dataset accuracy: **not-run**. CMake is not installed in this host shell;
recorded direct compiler commands exercised the test sources registered in CMake.
No hardware access attempted.

Remaining work includes complete native stage separation and behavioral audit,
source-aligned README coverage, YOLOE, B10/B11/H8, final host validation and fresh
independent whole-branch review. This increment fixes unsafe output handling;
it does not close the complete migration or claim native parity with Python.
