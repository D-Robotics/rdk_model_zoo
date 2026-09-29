# B10 ASR — UCP SDK adapter and preflight

2026-09-28. Author implementation record, Review=not-run, Closed=no. The native
CLI, vocabulary JSON loader and full native result reporting still need
integration. This does not close ASR or the full H0–H9 objective.

## Implementation

The native adapter reuses the existing packed-model/tensor owners and synchronous
UCP task helper. Preflight is mandatory and precedes every SDK initialization.
The production factory uses shared native identity and SHA-256 utilities to
verify exact S100/S600 identity, selected model bytes and the source-pinned
3503-token vocabulary. S100 + S100P board aliases are rejected. A locally
observed digest does not establish publisher provenance where the active
manifest has no expected digest.

Binding requires one named model, one unquantized FLOAT32 [1,30000] input and
one unquantized FLOAT32 [1,T,3503] output with positive T. Physical byte strides
and allocation sizes are checked before allocation. Integer, dynamic/missing,
misaligned, overlapping and undersized descriptors are rejected. Inputs are
copied through observed strides with padding cleared, caches are checked, and
outputs are compacted into independently owned raw logits. No decoding,
activation or file reading is added to the inference method. UCP scheduling
retains the existing ANY-core default without pretending to expose overrides.

The frontend and preflight now have a top-level static-library build. Vendor
SDK compilation is explicit (`ASR_BUILD_SDK=ON`) and requires actual headers and
libraries; it never falls back to the test double. On this host the expected
missing-vendor-header failure was observed and retained. Real SDK ABI/build
compatibility remains unverified.

## Shared resource finding and correction

The new failure injection reproduced an acquired tensor allocation leaking when
the allocation API returned an error with a nonnull address. The common
`OutputTensorOwner` previously tracked ownership solely from rc==0. It now
tracks the acquired address and also rejects rc==0 with a null address. Both
X5/UCP classification resource tests cover the additional cases. This matches
the existing common NV12 input owner's partial-allocation handling.

`initial-sdk-implementation.log` retains the failing resource-balance assertion.
After the fix, initialization, partial allocation, input/output cache errors,
task create/submit/wait/release failures unwind acquired resources in the host
API double. Destructor cleanup is nonthrowing; a real SDK release error cannot
be represented as a guarantee of successful resource reclamation.

## Validation and README updates

Evidence: [asr-sdk](evidence/2026-09-28-b10-asr-sdk/).

- Five ASR native tests pass, including actual audio libraries and API doubles;
  both direct tests and top-level library/test builds are exercised. Production
  adapter code is tested, not a reimplementation of its control flow.
- Descriptor fixtures use padded float input/output strides. Tests cover both
  targets, owned output surviving another call, malformed input, unsupported
  target and preflight failure before SDK calls, 18 constructor failure variants
  and six execution/cache/task failures.
- Twelve common native tests and eleven YOLOE native tests pass after the shared
  resource change. These include X5/UCP doubles; they do not run on hardware.
- 185 ASR/Ultralytics/checker Python regressions pass; migration contracts are recorded
  in `host-results.json`, with raw logs. Migration scope remains 47 samples,
  0 violations, 49 explicit policy skips and 0 exemptions.
- Six paired README shell/Python examples execute; 62 local links resolve. The
  complete bilingual C++ fixture example compiles and runs. Runtime guides now
  describe library options, SDK dependencies, identity/digest semantics,
  physical tensor validation, scheduling, ownership and the remaining CLI gap.
  The shared Ultralytics runtime guides also record the allocation fix.

Tests still do not certify vendor ABI, model metadata, real transcription,
board behavior, OE conversion or corpus CER. No board was contacted. Next work
is native CLI/model selection/vocabulary/report integration, then remaining
Paraformer/HIMLoco, B11, H8 and full-branch independent acceptance.
