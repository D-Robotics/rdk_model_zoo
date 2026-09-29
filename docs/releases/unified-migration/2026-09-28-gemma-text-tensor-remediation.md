# B11 Gemma Text tensor contract remediation — author record

Base: `6db93771` (materialized in the evidence directory). This is the
implementation record of the scoped Text tensor package plus the
GEMMA-TEXT-R1 and GEMMA-TEXT-R2 remediations from Codex's independent
review (below); re-review of the remediations is pending. Board execution
remains not-run and quantization recipes stay trusted, untouched source
material.

## Problem and change

Text prefill/decode bound their 35 inputs and 31 outputs without any
descriptor contract: tensors were allocated from raw SDK metadata, CPU input
preparation memcpy'd host buffers at assumed widths, logits argmax
reinterpreted storage as int16 unconditionally (`LogitsRowPtr`), KV outputs
were consumed through an unvalidated `stride[0]`, and sequence length was
taken from whatever the descriptor declared — so a differently exported or
corrupt HBM entered the engine silently. The fixed source's `FillCommonInputs`
wrote `seq_len` token ids into a 256-entry stack buffer; a 512-position export
overflows it. All of this is reproduced against immutable baseline source and
captured in the evidence directory (UBSan out-of-bounds index then ASan
stack-buffer-overflow in `FillCommonInputs`).

New `gemma4_text_tensor` (header + implementation) owns the fixed-export
contract and all raw physical IO for Text; `TextEngine` keeps only model
orchestration:

- Per-binding storage types are pinned — `inputs_embeds` F32 `[seq,1536]`,
  `token_ids` S64 / `position_ids` S32 one row of `seq`, masks S16
  `[seq,4096]`, logits S16 `[seq,262144]`, K/V inputs S8 dense
  `[4096,head_dim]`, K/V outputs S8 `[seq,head_dim]` — with quantization
  metadata rejected everywhere. Unknown or mismatched types are rejected;
  none is reinterpreted as float or at a guessed width.
- Singleton axes collapse before comparison (the source graph declares
  `[1,seq]` token ids and `[cache_len,1,head_dim]` caches), so one physical
  layout may be declared with or without them; anything else — extra batch
  axes, transposed or equal-element-count shapes — is refused.
- Byte strides must be element aligned, nonoverlapping, and every addressed
  byte stays inside the declared allocation and, via new `ModelIo` capacity
  tracking, inside the buffer the engine originally allocated. Output
  descriptors are revalidated after every inference against that capacity,
  not the refreshed claim. Logits argmax reads rows and columns through the
  descriptor; masks keep the source CPU int16 quantization (`round+clamp`,
  `kMaskValue`) and the argmax keeps `kLogitScale` scaling with first-max
  ties, so mask and sampling semantics are unchanged.
- Sequence dimensions are pinned to the export contract (prefill
  `kChunkSize`=256, decode 1); the stack buffers are gone in favor of
  seq-sized vectors. KV cache inputs must be dense (internal row padding is
  refused because `KvCache` owns raw contiguous matrices) and prefill/decode
  must agree on each cache allocation before one buffer is borrowed by both.
  Binding records the borrowed capacity, which must cover the declared
  allocation.
- Before every cache append, all 30 KV outputs are revalidated and gathered
  through `TextKvOutputRows`; per-layer K/V row strides must agree (the
  append advances both sides with one shared stride — previously only
  documented). `KvCache::ValidateAppend` now also rejects append sources
  pointing inside any resident K/V allocation, enforcing the documented
  separation prerequisite at the interface itself; no existing caller or
  behavior changes.
- Constructor validation runs before any allocation, so an incompatible
  export is rejected without acquiring buffers. The unused unchecked
  `LogitsRowPtr` helper is removed; public engine APIs are unchanged.

Ruling: the config constants (`kChunkSize`, `kCacheLen`, `kHiddenSize`,
`kVocabSize`, `kHeadDims`) are the export contract, matching the fixed
source graph (vocab 262144, head dims 256/512, sliding window 512). A
differently exported model needs an explicit adapter and must not silently
enter this fixed-layout engine. The full session-stage reorganization of
`TextEngine` beyond descriptor/IO boundaries is explicitly out of scope and
remains open.

## Checks

[Evidence directory](evidence/2026-09-28-gemma-text-tensor-remediation/)
contains the baseline failure, red/green driver, hashes, logs and this
contract summary (`checks.json`).

- Baseline reproduction: the 512-position fixture against immutable `git
  show` sources — constructor accepts, then UBSan `index 256 out of bounds
  for type 'int64_t[256]'` followed by ASan stack-buffer-overflow in
  `FillCommonInputs`; the same driver against current sources rejects at
  construction before any allocation.
- New `text_tensor_test` (helper contract): padding zeroing with row and
  leading-singleton layouts, compact-element-count and innermost-stride
  enforcement on writes, strided logits argmax including ties, negative and
  all-zero rows, per-role dtype/shape/stride/capacity/quantization
  rejections, dense-KV and KV-output rules.
- New `text_tensor_flow_test`: the production engine, cache and helper run
  against an SDK double speaking this contract — deterministic generation
  (prefill row 4 → token 104, decode rows → 100), cache transport verified
  through the borrowed KV inputs, inference-failure cleanup, post-inference
  logits/KV descriptor drift rejected, 13-case constructor rejection matrix
  with zero allocations, benchmark path.
- `kv_alias` joins the KV scenarios; `text_resources_test` now serves
  contract-valid descriptors and still passes 301 successive constructor
  failure points, six invalid descriptor cases and normal teardown.
- 14 native CTests pass under ASan/UBSan (bundled CMake, reused
  `../.coordination/gemma-vision-sanitized`, reconfigured); 29 Gemma sample
  unittest tests pass (new `test_cpp_text_tensors.py` compiles both new
  native tests from production sources); `samples/_shared/tests` passes 158
  tests; migration contract reports 51 samples / 0 violations / 51 policy
  skips / 0 exemptions.
- Bilingual runtime README gains the Text tensor transport contract table
  and test entries; quantization recipes are untouched.

## GEMMA-TEXT-R1 remediation (after Codex independent review)

Codex reproduced a defect in the new helper: `Canonicalize` rejected negative
but preserved zero dimensions, and `FlattenedElements` computed
`INT64_MAX / dimension[axis]`, so a descriptor with a zero dimension and a
positive allocation (F32/NONE `[0,1536]`, strides `[6144,4]`, aligned size
6144, `kInputsEmbeds` seq 256) aborted under UBSan
`integer-divide-by-zero` (`gemma4_text_tensor.cpp:120`, rc −6). The defect
had slipped through because the ordinary allocation-size check
(`alignedByteSize <= 0`) rejects most zero-dimension descriptors before the
shape path, and the sanitized build recovers from UBSan reports by default.

Fix: `Canonicalize` now rejects nonpositive dimensions before any
arithmetic, and `FlattenedElements` additionally treats a nonpositive
dimension as an impossible element count, so the division is unreachable
regardless of caller. Audit of the remaining helper arithmetic: the only
other division (`CheckStrides`) is guarded by a `steps > 0` short-circuit;
`DenseMatrix`, `ExpectMatrix`, `TextKvOutputRows`, `ArgmaxTextLogits` and
`WriteTextInput` perform no division on descriptor-derived values, and the
element-count overflow still returns −1 and is rejected.

Red/green proof, without modifying the reviewer evidence
(`evidence/2026-09-28-gemma-text-tensor-independent-review/`): the pre-fix
helper was materialized under `r1-before-fix/` (hash matches the review
record) and a 12-case driver (`zero_dimension_regression.cpp`) compiled with
the reviewer's sanitizer flags aborts with the same division-by-zero at the
same line (rc 134/−6, `r1-red-run.log`); the fixed helper rejects all ten
zero-dimension cases with `nonpositive dimension` while singleton-adjacent
positive layouts (`[1,256,1,1536]`, `[4096,1,256]`) keep validating
(`r1-green-run.log`, rc 0). The reviewer's own fixture, recompiled
unmodified, prints `rejected: … nonpositive dimension` with rc 0.

Permanent coverage: `text_tensor_test.cpp` gains zero
leading/singleton-adjacent/trailing/internal dimension cases across embeds,
mask, logits and KV roles — each with a positive allocation so the shape
contract itself must reject — plus the singleton-adjacent positive accepts;
`text_tensor_flow_test.cpp` adds a `zero_seq` constructor mutation to the
rejection matrix. The ownership fixture's `sequence` mode already drives the
engine constructor with zero dimensions and positive allocations under
sanitizers. Rechecks after the fix: 14/14 sanitized CTests, 29 Gemma sample
unittests, and the Gemma-scoped contract checker (0 violations) pass.

## GEMMA-TEXT-R2 remediation (after Codex independent review)

Codex reproduced an adoption exception-safety defect in the `ModelIo` this
package introduced: `AddInput`/`AddOutput` pushed the capacity entry before
adopting the tensor, so a `bad_alloc` in either vector growth leaked the
caller's buffer — a by-value `hbDNNTensor` parameter is a trivial POD that
cannot free its own buffer on the way out — and a failure between the two
pushes left a stale capacity entry desynchronizing `InputCapacity`. The
reviewer driver reserved the tensor vectors like production `InitModelIo`
does (the capacity vectors are not reserved) and observed `released=0` on
both input and output paths (rc 1, header `9e3d4abd…`).

Fix, keeping the public API and borrowed-KV ownership unchanged: `ModelIo`
now adopts through a transactional helper with an RAII guard. The guard
owns the incoming buffer; the capacity push and the tensor push complete as
one unit — a tensor-push failure pops the capacity entry and the guard
releases the buffer, a capacity-push failure leaves both vectors untouched
and the guard releases the buffer, and success disowns the guard exactly
once so no double free is possible. `BindBorrowedInput` is untouched: its
vector resizes precede any ownership change and the free/rebind tail is
non-throwing. `InitModelIo` stays as-is because capacity-vector growth is
now handled transactionally rather than needing reservations.

Red/green: the reviewer fixture recompiled unmodified against the
materialized pre-fix header (`r2-before-fix/`, hash matches the review
record) prints `input caught=1 released=0` / `output caught=1 released=0`
with rc 1 (`r2-red-run.log`); against the fixed header (`2516f13c…`) it
prints `released=1` on both paths with rc 0 (`r2-green-run.log`).

Permanent coverage: new `tests/native/model_io_adopt_test.cpp` (ctest
`model_io_adopt_test`, also compiled by `test_cpp_text_resources.py`)
replaces global `operator new` to inject a deterministic `bad_alloc` at
each adoption stage: capacity-push failure on `AddInput`, tensor-push
failure on `AddOutput` with the capacity rollback verified through
`OutputCapacity(0)`, successful adoption plus repeated `Clear`, and both
failure stages in sequence followed by a clean adoption. `hbUCPFree`
asserts a single release per buffer and the sanitized build passes with the
new-test-injected allocations intact. Run against the pre-fix header the
new test aborts at its leak assertion (`r2-newtest-prefix-run.log`, rc 134).

Rechecks after the fix: 15/15 sanitized CTests (the new entry included), 29
Gemma sample unittests, and the Gemma-scoped contract checker (0
violations). After Codex confirmed the R1/R2 fixes, the runtime README
count paragraph was corrected in both languages (fifteen CTest entries,
three Text ownership checks including tensor adoption under injected
allocation failure) to match the registered CTest list; documentation only.

## Status of related work (as of 2026-09-28)

- MiniCPM core refactor and H0 are already independently accepted; they are
  not pending on this package.
- Full Gemma session-stage reorganization is still pending: this package is
  tensor-contract only and is not H7 completion.
- The reviewer directory's `adoption_failure.cpp` /
  `adoption-failure.json` (GEMMA-TEXT-R2) is remediated above; the
  R1/R2 remediations themselves await Codex re-review and no H7 or
  session-stage closure is claimed.
- Board/vendor-ABI evidence, explicit model preparation and H0–H9 closure
  remain open items of the host-completion plan; this record makes no claim
  about them beyond the scope above.

## Boundaries and remaining work

No vendor SDK ABI, real HBM descriptor, model inference, board, download,
export/calibration or quantized-accuracy run was executed or revalidated.
Host doubles prove host code paths and pointer arithmetic only; whether the
published HBM's descriptors match this contract (including the pinned
262144 vocab and per-chunk KV output rows) remains board evidence. Session
staging inside `TextEngine`, explicit model preparation and the remaining
completion-plan items beyond the already-accepted H0 continue; see the
status section above for what is and is not pending.
