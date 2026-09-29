# Paraformer native S100 SDK adapter

Date: 2026-09-28. Implementation and host checks, not independent acceptance.
Base: `4d0b7b71dab7e80df1221b630099c913e9d97ea0`.

## Implementation and source contract

`runtime/cpp/src/sdk_runner.cc` supplies one synchronous raw model call per
`SdkRunner`, separate from CPU CIF/text and application composition. It loads one
artifact containing exactly one model and binds physical tensors by name, not
position. Encoder/predictor/decoder contracts match the pinned S source lookups
and unified Python `model_binding.py`. Decoder acoustic aliases and optional
`token_num` output are explicit. Count is int32; all other physical tensors are
unquantized float32. Unknown/duplicate roles, incorrect geometry/type/stride or
allocation are rejected.

Input sets are validated in full before buffer writes or SDK execution. Per-axis
byte-stride transfers support padded tensors; input padding is cleared. Results
are owned compact arrays, not reusable SDK views. Float inputs must be finite and
count is limited to 0–100. Raw output numeric validation remains in the consuming
pipeline. The class is serial-use; scheduling uses shared default priority/any
BPU core. Custom native scheduling is not claimed.

Packed-model, allocation and task cleanup reuse Ultralytics ownership/transport.
The mandatory preflight callback runs before SDK calls and must verify identity
and selected artifact. Concrete production preflight factory, three-artifact
selection and complete executable integration remain pending.

## Discovered shared transport restriction

The first integration test failed because shared `infer_sync` accepts only one or
two image input tensors, while decoder requires four. See `first-ctest.log`.
The implementation now extracts `infer_tensors_sync` for already validated raw
arrays with any positive input count. The existing image-facing `infer_sync`
keeps its one/two-input restriction and delegates task handling to the same
implementation. Tests cover both preserved rejection and successful four-input
submission/wait/release. No YOLO image protocol has been expanded.

## Verification

[Evidence](evidence/2026-09-28-b10-paraformer-sdk/):

- `red.log`: SDK implementation absent before adding code.
- `first-ctest.log`: reproduced four-input rejection before shared transport fix.
- `doc-0.log`: current Release build with ASan/UBSan, three Paraformer tests pass.
  SDK tests explicitly use an API double, not a vendor ABI replacement. They
  cover all three stage layouts, reordered/padded tensors, both acoustic aliases,
  optional count output, retained result ownership, invalid numeric inputs,
  duplicate aliases, count dtype, metadata rejection and resource cleanup after
  allocation/cache/inference/submit/wait errors.
- `doc-1.log`: complete numerical example returns `2 3 8`.
- `api-compile.log`: complete bilingual SDK embedding function compiles against
  the public header, without pretending to execute a model.
- `vendor-sdk-configure.log`: real SDK-enabled configuration correctly rejects
  missing vendor headers. This is an expected negative check, not a real SDK pass.
- `shared-ctest.log`, `asr-ctest.log`, `yoloe-ctest.log`: affected native regressions, respectively 12/12, 7/7 and 11/11 passed.
- `migration-gate.json`: migration-scope checker passed 47 samples / 0 violations / 49 policy skips /
  0 exemptions, distinct from complete sample
  acceptance. Paraformer remains outside completed migration scope.

Run the bilingual documentation check from repository root:

```bash
python3 docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-sdk/check_docs.py
```

Its exact commands and return codes are in `doc-summary.json`. The first version
of this checker expected an older CTest success summary wording and falsely
rejected a successful run; matching now accepts both formats while requiring
100% success for exactly three tests. No product gate was weakened.

## Documentation and remaining scope

Both native README files now document actual vendor build dependencies/options,
physical role names, types/strides/ownership, mandatory preflight responsibilities,
raw versus pipeline validation, scheduling limits and a compilable embedding
function. Root README and migration tracking continue to mark the complete native
entry pending. No customer-facing command silently substitutes a fake backend.

Actual SDK ABI/model metadata, HBM inference, board tests and OE are not verified.
Prepared-manifest/native CLI, concrete preflight, conversion/evaluator migration,
HIMLoco, B11/H8 and final whole-branch review remain open. B10 and H0–H9 are not
closed by this adapter implementation.
