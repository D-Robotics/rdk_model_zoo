# Paraformer host evaluator implementation record

Status: author implementation/host verification; not independent acceptance. Board,
real HMCT and real OE remain not-run. This does not close the whole sample or H0–H9.
Base: `510c9c337d05596c08f5d03f00ffb51e425385a4`; source S:
`380e1a2bf42041af54be6f34935e50197cfadff9`.

## Changes and source fidelity

The source `conversion/11_eval_pipeline.py` FP32 and HMCT paths now have a dedicated
`evaluator/main.py`. It reuses runtime pipeline/CIF/decoding and shared text metrics.
The extracted `bind_stage_io` validates raw ONNX without inventing a published HBM
selection; `bind_model` still validates publication identity and single-model SDK
metadata before delegating. Acoustic aliases, reordered interfaces and optional
int32 decoder count output are validated by name, never position.

The evaluator preserves HMCT's source executor/create_session/forward API. Its test
uses an explicit fake module, so the result is only adapter coverage. FP32 uses real
CPU ORT. Self-contained models are required; external weights are rejected rather
than left outside provenance. Stage values must match exact names/shapes/dtypes and
be finite. No quantized-output conversion is silently inserted.

Manifest validation includes all records before prefix selection. Selected features
are loaded and hashed from the same bytes, without coercion/pickle. Missing files,
wrong shape/type, invalid length, inconsistent truncation, digest mismatches and
trailing bytes fail. Unlike source, missing features are not skipped. Unlike the
old evaluator, decoding strips `@@`, matching the unified runtime; no repeat collapse.
Empty-reference CER is undefined/null instead of dividing by an invented character.
New output directories prevent overwrites. Failure retains partial utterances, error
and current ID but no aggregate metric. Successful completion rechecks model,
manifest and vocabulary hashes. Stage timings are not BPU/end-to-end measurements.

## Real FP32/source comparison

The already exported real-weight v4 models and two real prepared audio features
were evaluated; no model fixture substitutes for this run. Paths, SHA-256, interface,
versions and complete per-item predictions are in
[evaluation.json](evidence/2026-09-28-b10-paraformer-evaluator/fp32-evaluation.json).
The original source script was separately executed without source modification on
copies of the same features and the same ONNX models. It uses its original CIF and
CER code. [Source results](evidence/2026-09-28-b10-paraformer-evaluator/legacy-results.json)
and [comparison](evidence/2026-09-28-b10-paraformer-evaluator/comparison.json)
show both transcripts and per-item edit distances match:

| ID | Reference | Hypothesis | Edits |
|---|---|---|---|
| BAC009S0724W0121 | 广州市房地产中介协会分析 | 广州市房地产中介协会析 | 1 deletion |
| BAC009S0724W0168 | 新地王的诞生迅速搅热南沙土地市场 | 新地网的诞生迅速绞热南沙土地市 | 2 substitutions + 1 deletion |

Total: 4 edits / 28 reference characters = 14.285714% CER. Neither utterance is
exactly correct. This verifies flow/source comparison on two examples, not AISHELL
300-item accuracy, arbitrary-input equivalence or board output. The pre-existing
random-context Torch/ORT discrepancy remains disclosed in the export review.

Full raw logs: [new](evidence/2026-09-28-b10-paraformer-evaluator/fp32.log),
[source](evidence/2026-09-28-b10-paraformer-evaluator/legacy-fp32.log).
The local source output directory was `../.coordination/paraformer-legacy-evaluation-v1`;
new output was `../.coordination/paraformer-evaluation-v1`. Large models stay outside Git.

## Verification and documentation

75 sample tests passed, no skips, in the real Python 3.12 frontend/export environment:
[suite](evidence/2026-09-28-b10-paraformer-evaluator/sample-suite.log).
Ten evaluator tests cover legacy/prepared inputs, full-manifest validation, failure
retention, empty references, zero-token decoder bypass, named binding and HMCT adapter
protocol. Existing publication-binding/runtime/export/calibration tests also pass.

Both evaluator READMEs provide preparation, complete FP32/HMCT commands, all options,
input/output contract, normalization, historical metrics, source differences and
troubleshooting. Root/Python/conversion README pairs now link to that implementation.
The historical 300-item FP32/HMCT/S100 Python/C++ CER figures remain explicitly
historical; neither these tests nor two audio samples reproduce them.

Remaining: whole-sample structural/document review, real OE/HMCT external environment,
SDK/board validation and final independent branch review. Board verification remains
excluded from this host work at the user's request.

Additional checks: evaluator `--help` succeeds in the main host environment without
loading ONNX/HMCT sessions. All 130 local README links in this sample resolve (code
fences excluded). Migration-scope gate: 47 samples, 0 violations, 49 policy skips,
0 exemptions. Paraformer remains pending in that scope until its whole-sample
contract audit; this gate is not presented as its structural acceptance.
