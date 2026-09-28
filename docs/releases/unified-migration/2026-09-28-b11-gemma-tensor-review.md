# B11 Gemma Vision tensor contract and physical storage

Base `6a71bdaa`. Author implementation increment; independent/board acceptance remains open.

## Changes

Vision now validates semantic matrices before allocation: F16 [2520,768] input and F16/F32
[280,1536] output, with optional leading singleton axes and no quantization metadata.
Unknown/integer outputs no longer fall through to a float reinterpretation. Rank, dimensions,
nonoverlapping aligned byte strides, allocation span and original buffer capacity are checked;
output descriptors are rechecked after the SDK task refresh.

The dedicated `gemma4_vision_tensor` implementation writes F16 input using row and column byte
strides, zeroes padding, and extracts owned F32 features from F16/F32 physical output layouts.
NaN/Inf output and nonfinite/out-of-range prepared RGB inputs are rejected. Source input
float-to-half truncation is retained; this is runtime storage conversion, not quantization
recipe validation. Numeric format helpers and diagnostics have moved out of the inference
file. `VisionEngine::Infer` now composes tensor write, SDK execution and tensor read only.

The original source tutorial declares [2520,768] F16 patches and 280 image tokens; existing
runtime constants fix hidden size 1536. This host contract does not claim that a real deployed
HBM's metadata was newly inspected. That remains board evidence, excluded from this run.

## Checks

- The new standalone tensor test initially failed compilation on the absent implementation.
  It now checks known half encodings/subnormals, padded rows/elements, ownership, both output
  dtypes, invalid shape/type/quantization/stride/capacity, nonfinite values and null storage.
- Production VisionEngine is exercised with independent SDK doubles. They inspect the packed
  input and populate padded F16/F32 outputs, plus inject post-inference descriptor drift and
  invalid metadata. The pre-fix `6a71bdaa` path fails the independent input-padding assertion,
  captured using immutable Git source; the current implementation passes.
- 94 resource/transport scenarios pass ASan/UBSan across S100 (46) and S600 (48) compile branches.
- Three native CTests pass ASan/UBSan, including the four source-image byte-exact preprocessing
  comparisons and the stage ownership checks. Gemma unittest discovery passes 12 tests.
- Migration contract: 50 samples / 0 violations / 51 policy skips / 0 exemptions.
  Bilingual runtime README includes physical contracts and updated source navigation.

[Cases, compiler commands, source hashes and baseline failure](evidence/2026-09-28-b11-gemma-tensors/checks.json),
[migration contract](evidence/2026-09-28-b11-gemma-tensors/migration-contract.json).

## Boundaries and remaining work

No real vendor SDK ABI, model inference, board, quantization, download or compilation recipe
was executed. The tests prove host code paths and pointer arithmetic, not BPU behavior or
performance parity. The Text/KV engine's contracts, state and resource ownership remain to
be refactored, as do explicit model preparation, MiniCPM and final H0–H9 acceptance. Gemma
and B11 remain in progress; the previous transport findings are addressed within this host scope.
