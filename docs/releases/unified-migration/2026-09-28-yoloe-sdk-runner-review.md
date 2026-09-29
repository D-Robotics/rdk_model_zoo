# YOLOE native SDK adapter: owned resources and float outputs

Base: `7c4e5b3e730cecbf19d1819848798b0e1d2f4b59`. This increment implements
the SDK adapter library; the canonical preflight policy and CLI remain pending.
H0–H9, other migrations and whole-branch independent review remain open. This
is implementation self-check, not independent acceptance or board validation.

## Implementation

`SdkRunner` implements the existing Runner interface. It validates the explicit
target/variant against the compiled SDK stack, runs a mandatory application
preflight before SDK access, checks the file and loads one named model. Input
must be target-specific 640×640 NV12; all ten output roles must have validated
physical NHWC FLOAT32/no-quantization descriptors before any tensor allocation.
S published quantized outputs remain rejected. Shape matching cannot prove
weight/vocabulary identity; the mandatory policy boundary must do that work.
No default/no-op production gate or automatic publication selection is supplied.

The adapter reuses shared PackedModelOwner, Nv12Input, TaskOutputs and infer_sync.
TaskOutputs now accepts a bounded expected count and semantic validator while
its existing pose/segment overload delegates unchanged. There is no parallel
YOLOE tensor allocator or manual dequantizer. Physical outputs are copied into
independent vectors and moved into canonical role order. Destruction releases
outputs, inputs and then the model, including partially constructed states.
The existing yoloe.cpp remains only construction/pre/infer/post/predict.

The SDK library is opt-in through YOLOE_BUILD_SDK. Matching DNN include/library
paths are required; UCP needs its matching library too. The pure stage library
stays usable without vendor SDK headers. Host fixtures compile production
adapter and shared I/O against narrow API doubles; they are not ABI evidence.

## Verification and documentation

The first test compilation failed because sdk_runner.h did not yet exist;
that log is preserved. Eight native tests pass, including both SDK stack
branches under ASan/UBSan. Adapter cases cover semantic reordering, output
independence across calls, exact input lengths, preflight-before-SDK ordering,
quantized/duplicate/missing output rejection, partial allocation, initialization
failure with acquired handle, input cache and inference errors, nonfinite
output rejection, one-model requirement and missing file rejection. UCP E11s
construction is also covered; E26 uses UCP and E11 uses X5 for inference tests.

Bilingual README now documents the adapter API, concrete resource lifetime,
float-only boundary, exact allowed pairs, mandatory policy responsibility,
optional SDK build and absence of a deployable CLI. Both complete API functions
are compiled, host build/test blocks are executed, and the SDK configure path
is checked for a clear missing-dependency error. Real SDK build/run, board
inference, compiled float S assets, OE acceptance and dataset accuracy remain
not-run/unverified. No synthetic fixture result is labeled board inference.

Commands, full outputs, timestamps and hashes are retained in
[evidence](evidence/2026-09-28-yoloe-sdk-runner/).

Final verification: both documented YOLOE host build modes passed all eight
CTest cases; the shared Ultralytics CMake suite passed all 12 cases. Both API
functions compiled with C++17 warnings treated as errors. All 463 Python tests
passed (29 YOLOE, 143 Ultralytics, 153 shared, 52 ResNet, 44 OCR, 27 checker,
5 export, 10 evaluator). Migration check: 45 samples, zero violations,
47 declared policy skips and zero exemptions; README link/API checks passed.
The missing real SDK configure check returned the expected error explicitly
naming the DNN headers/library, and is not counted as a successful SDK build.
