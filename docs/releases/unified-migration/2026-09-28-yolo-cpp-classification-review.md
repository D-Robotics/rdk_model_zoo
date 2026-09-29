# Ultralytics C++ classification output binding

Implementation self-check, 2026-09-28, base `cb56892b0a86e42c431c360f157a75f1c038e8a1`.
H2/native audit remains open; this report is not final independent approval.

## Defect and changes

The previous classification main read `validShape.dimensionSize[1]` as the class
count, cast any output allocation to consecutive floats and partially sorted five
entries without a lower size bound. A `(1,1,1,1000)` tensor could therefore cause
an out-of-range Top-K operation; padded class axes read unrelated values. Output
type and quantization were not validated, allocation/cache failures were ignored,
and early returns leaked packed model/output allocations.

The new pure `common/classification.h` validates a single 1000-class vector,
physical class stride and allocation bounds before reading. It copies only valid
finite FLOAT32 logits, calculates stable softmax using double accumulation and
returns owned Top-K records, sorting exact probability ties by ascending ID.
`classification_binding.h` adapts X5 aligned dimensions and S byte strides;
integer/SCALE, batched/spatial/other-class outputs and unusable allocation sizes
are rejected. No manual dequantization path is added.

`dnn_resources.h` owns packed models/output tensors; acquired resources release
on normal return, SDK error return or C++ exception. Allocation/cache errors stop
the main pipeline. Model count and name pointers are checked before indexing.
The 1000 source labels are moved verbatim to `imagenet_labels.h`; dead NV12 code
and private softmax/Top-K helpers are removed from main. Input handling continues
to use the shared `Nv12Input` and synchronous inference wrapper.

The existing C++ letterbox/127 preprocessing default is deliberately preserved.
The bilingual README explicitly distinguishes it from Python YOLO26/S stretch
classification; this change makes no cross-language accuracy equivalence claim.
It documents shape/stride requirements, tie ordering, errors, ownership and
actual validation limits. Two stale Python README sentences about OBB auditing
were updated to point to the now-documented stages.

## Validation and discovered test defect

- [Native command records](evidence/2026-09-28-yolo-cpp-classification/native-results.json):
  seven C++11 executables compiled and ran with AddressSanitizer and UndefinedBehaviorSanitizer;
  all 14 compile/run commands returned zero. Five test pure production helpers;
  two compile production classification binding/resource code against **narrow
  descriptor test doubles**, not actual X5/UCP headers or libraries.
- Classification covers singleton layouts, NHWC/NCHW vector axes, padded reads
  with NaNs in padding, overflow/underallocation, dtype/quantization refusal,
  nonfinite logits, extreme softmax, deterministic ties and invalid Top-K.
  Resource tests cover acquired-resource cleanup on exceptions and allocation failure.
- The sanitizer run found an existing **test fixture** overflow: `test_decode.cc`
  allocated 16 bins but passed them to a four-side/64-value DFL decoder. The
  fixture now allocates and initializes all four sides and checks all four results.
  Its preserved raw ASan log includes one trailing-space diagnostic line; the
  maintained-file whitespace check excludes raw `.log` files.
  The [initial ASan failure](evidence/2026-09-28-yolo-cpp-classification/test_decode-initial-asan.log)
  and initial command records remain preserved. Production DFL decoding was unchanged.
- [Host records](evidence/2026-09-28-yolo-cpp-classification/host-results.json):
  Ultralytics 141, shared 144, ResNet 52, OCR 44 and checker 27: 408 tests passed.
  Migration contract check: 44 samples, 0 violations, 46 policy skips, 0 exemptions.
  128 local README links and 16 executable bilingual Python examples passed.

The CMake test list now includes the new tests, but CMake itself was unavailable
in this host shell; native checks used the recorded direct compiler commands.
Full classification executable build with real OpenCV/DNN SDK, SDK ABI/cache
behavior, board inference and ImageNet dataset evaluation are **not-run**.
These are not inferred from pure-helper or descriptor-double success.

Remaining full scope: detection/pose/segmentation native audit and refactoring,
README/source-capability audit, YOLOE, B10/B11, H8 repository integration, full
host gates and fresh independent whole-branch review. H0–H9 remain open.
