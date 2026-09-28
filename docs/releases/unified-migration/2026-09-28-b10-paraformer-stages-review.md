# Paraformer public stage API correction

Base: `7f1a3e123a4f2bfa07e2b1065ce1ef219a1a6f7f`.
Status: author implementation/host verification. Corrects the API finding in the
[README review](2026-09-28-b10-paraformer-readme-review.md); not independent acceptance.

`runtime/python/stages.py` exposes EncoderStage, PredictorStage and DecoderStage.
Each has public pre_process/forward/post_process methods. Pipeline construction
publishes these as encoder_stage/predictor_stage/decoder_stage while retaining
existing injected raw runner attributes and constructor arguments. There is no
SDK loading, CLI, metrics, file handling or next-model execution in these tasks.

Preprocessing returns owned physical tensors. Decoder also snapshots the token
count in PreparedInput.context, rather than keeping it in mutable stage state.
Forward validates inputs, calls one runner and validates/copies raw outputs without
activation, CIF or text decoding. Encoder postprocessing returns context; predictor
returns weights/hidden; decoder decodes using explicit count context. CPU CIF stays
between predictor and decoder in predict; zero tokens still bypass decoder. Direct
decoder forward rejects counts outside 0–100 even if pre_process was bypassed.
Physical dtype/name/shape contracts and optional decoder count remain explicit.

StageError is a ValueError carrying stage/operation and chaining the original
exception. Raw transport and malformed outputs are attributed to the corresponding
stage; CIF errors use cif/integrate. Failures stop later model execution. SDK/thread
concurrency is not inferred from per-call state ownership; callers must serialize
access unless their injected runner guarantees otherwise.

## Verification

- The original five added tests failed before the stage implementation existed
  ([red](evidence/2026-09-28-b10-paraformer-stages/red.log)). That initial invocation
  also rediscovered the imported six baseline test methods; the test module import
  was corrected so the final suite contains no such duplicates.
- Final suite: 82 tests, no skips. Seven new stage tests cover explicit execution
  vs predict, all-three-stage raw-value purity, A/B/A context isolation and owned
  buffers, transport cause chaining, malformed output attribution, CIF failure
  short-circuit and direct decoder-count validation.
  [Full suite](evidence/2026-09-28-b10-paraformer-stages/sample-suite.log).
- Real FP32 evaluation reran both prepared audio inputs through all three actual
  exported models. Text, every selected token ID, count, decoder-executed flags
  and full CER metrics are identical to the earlier evaluator record. Timings are
  excluded from equality checks. CER remains 4 edits/28 chars, 14.2857%, not a
  dataset benchmark. [Full result](evidence/2026-09-28-b10-paraformer-stages/fp32-evaluation.json),
  [raw log](evidence/2026-09-28-b10-paraformer-stages/fp32.log),
  [comparison](evidence/2026-09-28-b10-paraformer-stages/comparison-and-examples.json).
- English and Chinese README code blocks execute both predict and explicit stages,
  asserting text/token equality. Both were extracted and run with real Python.
  Physical contracts, ownership, error API and timing scope are documented.
- Direct sample contract: 0 violations, 1 CLI policy skip, 0 exemptions.
  [Result](evidence/2026-09-28-b10-paraformer-stages/contract.json).

Forward timings now include stage validation/copy and adapter execution. They are
not accelerator-only latency. No perf comparison is made. Real OE/HMCT/SDK/board
execution remains not-run; the random-hidden-state Torch/ORT discrepancy in export
is not waived by this unchanged two-utterance result.

Paraformer now enters the ongoing migration checker scope as in-progress. Its
source-to-unified whole-sample audit and independent acceptance remain open;
H0–H9 are not closed by this stage correction.
