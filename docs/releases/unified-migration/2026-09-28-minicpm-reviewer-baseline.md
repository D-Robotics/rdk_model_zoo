# MiniCPM reviewer baseline for Claude Code / GLM implementation

Reviewer: Codex. Base `0aff05a8acc94553a2f6f628c578cf5823be7e0c`.
This reviews the inherited native core before the delegated implementation, not a
completed Claude submission. User now assigns implementation to Claude Code/GLM
and independent review/direction/GitHub synchronization to Codex.

Three findings reproduced with small independently defined host SDK doubles.
No production implementation was edited. No actual SDK, model, quantization or
board was run. Evidence and the exact fixture sources are under
[evidence/2026-09-28-minicpm-reviewer-baseline](evidence/2026-09-28-minicpm-reviewer-baseline/).

## R1 — legacy request state survives reinitialization (P2)

`runtime/legacy/src/minicpm5.cc:17–32,34–56`: predict destroys the handle and nulls
it, allowing init again, but ended_/failed_ are not reset. The double emits END
only for the first request. Both first and second predict return success and print
ended=1, despite the second request never receiving END.

Close by enforcing a clear single-use lifecycle (reject reinitialization) or by
resetting request state before a deliberately supported new request. Do not silently
introduce conversation/multi-request capabilities beyond the documented source.
Regression must include first success followed by missing-END and error behavior.

## R2 — S600 accepts nonfinite/negative metric values (P2)

`runtime/cpp/src/minicpm5.cc:79–88`: response presence is checked, but returned
metric values are copied without validation. The double returns NaN TTFT and -1
decode_tps; Generate accepts both. JSON presentation may turn nonfinite values into
null, losing the distinction between a successful result and invalid measurements.

Close with explicit finite/nonnegative metric handling consistent with the public
Result contract and CLI output. Preserve valid zero throughput for the documented
one-token length-limited case. Include separate invalid TTFT, decode and e2e cases;
do not simply reject every zero or silently coerce invalid values into zero.

## R3 — runtime configuration is not exception-safe (P2)

`runtime/cpp/src/minicpm5.cc:34–45`: remove(filename) occurs only after Init returns.
An injected C++ exception from Init leaves one minicpm5-* temporary file behind.
An exception from JSON serialization after mkstemp also lacks an owning guard;
that second path is static analysis, not a reproduced allocation failure.

Close with RAII for the temporary descriptor/file, covering success, SDK error
return and exception paths. Keep configuration/file IO outside the inference-stage
file during the requested responsibility refactor. Tests should verify cleanup
without assuming the real vendor API throws in normal operation.

## Reviewer expectations for this package

Preserve native generation parameters, S600 two-turn behavior, legacy streaming and
status semantics. Template/file handling, console output and metrics serialization
belong outside inference. Expose useful actual SDK-stage boundaries rather than
inventing tokenization APIs the SDK does not expose. Match final bilingual API
examples to implemented interfaces; keep all source quantization/evaluation content
and historical accuracy failures. Author tests do not equal independent approval.

Reproduction used C++17 with production minicpm5.cc, fixtures first on the include
path and the already-present nlohmann/json header at
`../.coordination/asr-json/include`. Drivers `legacy_reuse.cpp` and
`s600_boundaries.cpp` accept a temporary template path and temporary directory/case
respectively. `metric` and `init_exception` select S600 cases. Each returned 1 to
signal reproduction of the defect; stdout is persisted in findings.json. Fixture
model files contain only the word fixture and are never interpreted as weights.

## Navigation follow-up — verified 2026-09-28 at 3d1afeb0 (P2)

R4: Root English/Chinese README and samples indexes still state 50 native samples
and one LLM. The current four-category inventory is 45 vision + 3 speech + 1 robotics
+ 2 LLM = 51. None of those four files links MiniCPM, so the new sample is absent
from the customer/Agent discovery path. Add bilingual MiniCPM links and update
counts while retaining its in-progress status; do not imply native core acceptance.
Gemma model-preparation wording is also stale against the completed preparation
commits and should distinguish preparation completion from core work still open.

R5: `docs/release/s/models.yaml` already registers MiniCPM's three pinned archives
and uses `samples/llm/minicpm5-2b` paths, but its note says those paths still retain
the source-branch layout until B11 relocation. Update the explanatory note to the
actual migration state, preserving all archive URLs and digests. This is not a
request to download or validate the archives. Evidence: navigation.json.

These additions belong to the delegated MiniCPM package's allowed index/manifest
synchronization scope. They are reviewer instructions, not direct implementation
changes by Codex. Skills pack files were observed to exist (seven SKILL.md files);
H8 must verify their integration rather than assuming they are missing or blindly
recopying them from upstream. Existing pack provenance reports remain historical.
