# Paraformer independent host runtime review

Reviewer: Codex. Base `f09b78aa`. Accept the inspected host runtime composition,
public Python model-stage APIs and corresponding usage/contracts. No blocking
finding identified in this bounded review. Whole sample source/document audit,
B10 and full integration retain separate acceptance scope. No implementation was
changed by the reviewer.

Reviewed Python stages, application pipeline, raw runner construction/scheduling,
CPU CIF, greedy decoder and frontend interface; native composition/contract
interfaces; paired Python runtime descriptions and the native integration guide.
The encoder/predictor/decoder remain separately bound raw model executions. CIF
is explicit between models, never hidden in a forward transport. Prepared arrays
and returned context own storage; decoder count is per call. Stage errors carry
operation and original cause, and failed stages do not continue to another model.

The fixed contracts preserve [1,400,560] features, predictor weights/hidden, capped
100-token acoustic embeddings and 8404-class decoding. No-token CIF bypasses the
decoder, with absent rather than fabricated zero timing. Greedy Paraformer text
removes special/BPE markers without applying ASR CTC repeat collapse. The CIF
one-fire-per-frame source convention and valid-frame masking are documented.
Native composition exposes explicit raw callbacks and CPU bridge functions;
Python public stages are not misrepresented as identical native method names.

Frontend inspection confirms pinned CMVN, explicit 16 kHz input, source FunASR
configuration and scoped CPU RNG restoration. Documentation distinguishes prepared
features from audio, truncation from full-utterance processing, host frontends from
board SDK, and native prepared-feature input from Python audio processing. Its
complete stage examples, model identity groups and timing explanations describe
the current API. Historical frontend/FP32 author comparisons remain historical;
this review does not replace them with a new model-accuracy claim.

## Independent verification

[Runtime tests](evidence/2026-09-28-paraformer-independent-review/runtime-tests.json):
46 tests pass across binding, pipeline, CLI, stages, CIF, frontend input contracts,
mocked model preparation and native launcher/report validation. The download tests
use fixtures; no model was acquired. Conversion/export/calibration/evaluator model
execution tests were not selected. Candidate runtime hashes were rechecked before
this report and all remain identical.

[Native checks](evidence/2026-09-28-paraformer-independent-review/native-tests.json):
existing ASan/UBSan host configuration rebuilt successfully; six CTests pass for
contract, pipeline, SDK fixture, preflight, prepared features and CLI fixture help.
Sample checker has zero violations, one CLI skip and zero exemptions. The CLI help
CTest alone is not an end-to-end recognition test; Python launcher tests validate
report acceptance separately. No new real frontend environment or vendor SDK was
installed, and no full linked board executable was certified.

No board, new transcription/accuracy measurement, actual weight download,
export/calibration/OE/HMCT or quantization validation was performed. User-trusted
recipe text is not blocked on those executions. The full source README-depth audit
and final integrated checks still need their own evidence before global closure.
