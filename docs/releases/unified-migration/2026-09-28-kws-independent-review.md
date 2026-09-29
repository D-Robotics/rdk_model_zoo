# KWS independent host review — runtime accepted; README overview follow-up

Reviewer: Codex. Base `0d622a88`. Inspected the task, frontend, binding, raw runner
adapter, scoring helper, offline metrics/CLI and hierarchical customer guides.
No blocking implementation finding was identified in that host scope. This does
not close all B10 samples or the complete README quality audit.

The task has three stages plus predict; file loading, feature extraction, scoring
and SDK ownership are separate. The fixed frontend consumes mono finite float32
16 kHz samples, pads/truncates to 60000 (3.75 seconds), and expects 373x80 features.
It rejects unsupported configuration rather than reshaping incompatible features.
The source PaddleAudio call remains lazy; no alternate approximation was silently
substituted. Raw inference returns owned outputs; probabilities are validated and
reduced by maximum without an extra sigmoid. Exact S100 asset selection and board
identity checks precede actual SDK loading. Other targets are not silently reused.

The standard-library evaluator validates record IDs, labels, scores, threshold
and declared provenance. Its confusion counts and undefined-denominator policy
match the guide. Rates describe clips/windows, not false accepts per hour. Saved
hashes bind inputs but do not authenticate claimed model or dataset provenance.
No fabricated dataset score or new board measurement appears in the reviewed
root/runtime/evaluator text. Current commands explicitly separate preparation
from inference, and historical source latency/confidence retain attribution.

Independent [verification](evidence/2026-09-28-kws-independent-review/verification.json):
13 tests pass; checker zero violations, one CLI skip, zero exemptions. Tests use
injected frontend/SDK fixtures, not real BPU execution. Existing author evidence
for three actual PaddleAudio frontend cases is separately linked in
[the migration report](2026-09-28-b10-kws-review.md); that historical author run was
not repeated or relabeled as this independent run. No dependency was installed.

## KWS-N1 — restore an in-place algorithm overview

The current root guide says algorithm context remains in the archived S snapshot.
The pinned S root includes an MDTC algorithm overview and framework context. The
user explicitly requires systematic canonical guides with source explanatory
depth; archive navigation alone does not satisfy this. Restore a concise meaningful
algorithm/pipeline explanation in both root READMEs, including what the temporal
model consumes and how clip-level confidence relates to its output. Preserve
useful source context with attribution; do not strengthen source marketing claims
about robustness/accuracy or assert unverified dynamic-weight behavior as a
measured fact. Keep the corrected 3.75-second window, exact target support,
current commands and historical evidence boundaries intact. No code or recipe
execution is needed. This bounded documentation follow-up remains open; runtime
host review and whole-sample documentation acceptance are separate dispositions.

## KWS-N1 closure (2026-09-28)

Closed after independent reading of the restored bilingual root overview and
comparison with fixed-source description and current frontend/scoring modules.
The algorithm claims remain source-attributed; 60000 samples = 3.75 seconds,
[1,373,80] fbank and max-probability threshold decision are stated directly.
Commands and runtime code are unchanged. See [final documentation review](2026-09-28-readme-depth-special-independent-review.md).
