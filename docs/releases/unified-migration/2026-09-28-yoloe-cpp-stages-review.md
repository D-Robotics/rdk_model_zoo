# YOLOE native stage library and owned NV12 input

Base: `84187b0469616955f938c4d28b352440ce058f19`. This increment connects the
native math into a reusable stage library. Concrete SDK runner/identity gates,
CLI and board build/run acceptance remain pending. B9/H5/H0–H9 remain open.
This is implementation self-check, not independent whole-branch review.

## Architecture and behavior

`src/yoloe.cpp` contains construction and only pre_process/infer/post_process/
predict orchestration. Image operations, NV12 conversion, configuration,
transport types and numerical result assembly are in separate modules.
`common/nv12.h` uses actual OpenCV BGR-to-I420 conversion and reuses Ultralytics'
`i420_to_split_nv12`; it does not duplicate plane interleaving. Compact Y and UV
vectors own 409600 and 204800 bytes. Future SDK tensor strides are a backend
concern; the stage library does not pretend compact vectors are padded device
allocations.

The task owns a unique backend. Construction rejects null/protocol-mismatched
backends and invalid family-specific options. Prepared and raw batches carry a
private instance identity and immutable-access geometry; cross-instance use is
rejected, even for matching protocols. Raw tensors and returned ROI masks own
storage, so subsequent inference cannot invalidate previous results. Inference
calls the runner once; preprocessing/postprocessing never invoke it. No stage
loads files, downloads assets, selects hardware or renders images. Predict only
composes stages. One task/backend per inference thread remains the contract.

The Runner interface requires independently owned semantic FLOAT32 arrays and
puts real SDK metadata/artifact/hardware validation on the concrete backend.
There is no real SDK adapter in this increment. A fixture claiming a protocol
is not evidence of hardware validation; README examples explicitly require an
application-supplied matching backend.

## Host verification

Six native tests run with ASan/UBSan, including a stage test first compiled
before its API header existed (missing-header failure) and then implemented.
The stage fixture checks exactly-one backend call, manual stages versus predict,
original-image independence, explicit per-image geometry, cross-instance
rejection, propagated backend errors, malformed output rejection, configuration
errors, and backend destruction when construction fails. The runtime fixture is
synthetic and is never reported as SDK inference.

Six full NV12 comparisons use the actual bundled office image and a fixed-seed
333×1000 image, each with E11 letterbox/stretch and E26 letterbox. Geometry and
all Y/UV bytes agree with canonical Python. Complete plane arrays, hashes,
commands and timestamps are retained in [evidence](evidence/2026-09-28-yoloe-cpp-stages/).
The comparison probe rejects any attempt to invoke inference. This verifies
input preparation only, not BPU results or NV12-roundtrip model accuracy.

The native directory now has a standalone `yoloe_core` CMake library target.
The tests can also still be configured separately. Both bilingual README build/
run blocks and the complete C++ API function are compiled in host verification.
No OpenCV wheel is mistaken for a C++ development installation; the task-local
real OpenCV 4.14.0 build is reused. The external OpenCV library itself remains
unsanitized while sample/library code is instrumented.

Final recorded results: standalone and library builds each passed all six native
tests; the API example compiled with warnings treated as errors; all six NV12
comparisons and 29 YOLOE Python tests passed. Migration contracts checked 45
samples with zero violations, 47 declared policy skips and zero exemptions.
Full commands, timestamps, output and implementation hashes (including the
reused NV12 helper) are in the linked evidence directory.

## Remaining work

SDK allocation/cache/submit/wait/release ownership, target/variant/asset identity,
CLI integration and corresponding real SDK/board verification remain open.
Other samples, top-level/subdirectory README quality and final independent
whole-branch acceptance remain part of the active objective.
