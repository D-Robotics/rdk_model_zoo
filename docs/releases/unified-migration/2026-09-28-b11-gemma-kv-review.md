# B11 Gemma KV cache allocation and state

Base `07eee93c`; author host implementation record, not model/board acceptance.

## Findings and fixes

The source cache used the K allocation length to reset V, producing a heap overrun when V
was smaller. Reallocation freed the old buffers before the new allocation sequence completed,
so failure could invalidate aliases and leave partial state. Both failures are reproduced
against immutable baseline source under host allocator doubles and captured in the evidence.

K/V capacities are now separate. Allocate requires 15 valid sizes for each side and uses a
fully owned candidate cache; success swaps allocations and resets position metadata, while
failure destroys only candidate resources and preserves old addresses/data/state. Reset clears
each buffer using its own capacity and preserves aliases. Cache rolling and prefix retention
operate on the logical S8 [4096, head_dim] matrix, not trailing allocation padding.

Append validates all layer pointers, source row strides, positive 1–256 chunk size, sequential
position and arithmetic before modifying rows. Potentially allocating position bookkeeping is
prepared first, so allocation failure cannot leave partially advanced positions. Decode reuses
the same one-row append path; retired positions normalize to -1. Layer access is bounds checked.

Ruling: preserve the source implementation's prefix-retention behavior, correcting its header
promise of middle-range deletion. CompactShift retains the first resident n_keep rows and
discards the suffix; discard must equal the old logical length minus n_keep. TextEngine's
existing ContextShift uses exactly that form. Cost: external callers relying on the misleading
header must replay their retained suffix explicitly rather than expecting it to remain.

Ruling: remove unused SetOccupiedLen, which could desynchronize rows and positions without
updating the mapping. No active in-repository caller uses it. Cost: external library callers
must use Reset/Append/CompactShift to maintain state instead of overwriting a counter.

## Verification

- Six new native scenarios: asymmetric K/V reset; partial allocation failure; null allocation
  success; invalid size vectors/layer indices; append/late invalid pointer/prefix retention;
  successful reallocation resetting position state.
- Nine native CTests (three previous Vision checks plus six KV scenarios) pass ASan/UBSan.
- Gemma unittest discovery passes 13 tests, including the new host C++ KV group and previous
  launcher/resource/tensor tests. The earlier 94 resource/transport scenarios remain covered.
- Migration contract remains 50 samples / 0 violations / 51 policy skips / 0 exemptions.
- Bilingual runtime guides now distinguish allocation from Reset, logical positions from
  physical rows, alias lifetime, tail padding and actual CPU row movement.

[Baseline failures, current hashes and test output](evidence/2026-09-28-b11-gemma-kv/checks.json).

## Limits and next work

No model/quantization/board operation ran. Append's raw-pointer interface cannot discover source
buffer lengths; callers must supply disjoint readable output buffers and equal K/V output row
strides. Text SDK descriptor validation must enforce those conditions. Text ModelIo/engine
constructor cleanup, tensor bindings, mask/logit/session boundaries, explicit model preparation,
MiniCPM and the full H0–H9 goal remain open. This is a KV component improvement, not Gemma closure.
