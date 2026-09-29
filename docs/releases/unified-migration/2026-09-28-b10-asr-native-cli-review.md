# B10 ASR — complete native entry and customer guides

2026-09-28. Implementation/self-verification record; independent Review=not-run,
Closed=no. Native integration is implemented, but actual vendor SDK compilation,
ABI/model behavior and board execution are not verified. Full H0–H9 remains open.

## Final implementation

The native CLI now composes preflight, fixed-vocabulary JSON loading, SDK binding,
real audio reading, three-stage inference, complete-file chunk concatenation and
result recording. The task class remains limited to construction/configuration
and pre_process/forward/post_process/predict; parsing, file I/O and JSON saving
live outside it. JSON uses nlohmann 3.11.3 in the host test environment, matching
the source's dependency family; the header is not vendored into the sample.

Vocabulary bytes are hash-checked before parsing and mapped into exactly 3503
ordered IDs. Direct CLI options reject unknown, duplicate, missing-value and
cross-target asset identities. Required model SHA is supplied by the launcher
after active-manifest file validation. Model, audio and vocabulary are checked
again at completion. Missing publisher SHA is not promoted into verified origin.

Both success and failure records identify the execution backend. The native CLI
saves full metadata and per-chunk geometry/text; a later failure retains completed
chunks in failed.json. A new output directory is mandatory. The launcher retains
raw stdout/stderr, exact argv/cwd, UTC intervals, return codes and file digests.
It rejects a child that exits successfully but emits a fixture/invalid report.
CMake's real CLI requires the real SDK; host-fixture compilation is a separate,
explicitly named test target and cannot silently substitute for it.

## Regression findings fixed during implementation

- A nonzero native child was initially recorded as not executed because an error
  was raised before setting the flag. The log helper now marks execution when
  the child returns, including nonzero results; raw invalid-UTF8 stdout is retained
  byte-for-byte while terminal display may use replacement characters.
- Report validation initially accepted a boolean input dimension and an effective
  target length larger than the source chunk can produce. Explicit integer-shape
  checks and resampling-capacity bounds now reject both. Initial failures are
  retained in the evidence directory.

## Validation

Evidence: [native-cli](evidence/2026-09-28-b10-asr-native-cli/).

- Seven native tests pass with real audio libraries and explicitly marked SDK
  doubles/CLI fixture. The existing resource/stride/preflight tests remain active.
- Sixteen full native/launcher cases pass: S100 CTC/legacy, S600 CTC, board/asset/
  digest/vocabulary mismatch, missing model, empty/nonfinite audio, existing
  output, duplicate/unknown options, injected failure after one chunk, and public
  launcher rejection of the host-fixture report. Raw outputs and JSON are retained
  under the timestamped fixture-runs directory. No compiled model is used.
- The real bundled 73440-frame recording is processed as 30000/30000/13440 frames.
  Fixture CTC text is AAAAAA; legacy text is AAAAAAAAA. These are deliberately
  artificial decoder results, not recognized Chinese speech or model accuracy.
- Twenty-one ASR Python tests cover transport, frontend/decoding, CLI resolution,
  identity gate before build, report validation and launcher success/failure using
  mocked child reports. A mocked native-sdk label is unit data, not SDK evidence.
- Five bilingual shell/Python examples execute, 66 local links resolve, and the
  complete bilingual C++ task example compiles/runs. Root, model and native guides
  now agree that the entry is implemented and real SDK behavior is unverified.
  All launcher/direct binary arguments, defaults, build dependencies, failure
  outputs, lifecycle and Python/native resampling differences are documented.

The source's historical measurements remain in evaluator documentation with
scope intact. No conversion checkpoint/compiler recipe, corpus CER, real model
score or BSP compatibility result has been invented. Related regressions total 456 Python tests, with 121 publisher tests passing.
Full migration checks cover 47 samples, 0 violations, 49 policy skips and
0 exemptions. Raw logs and commands are retained alongside this report.

## Remaining full-goal work

ASR awaits fresh whole-branch independent acceptance and actual SDK/model checks
when an appropriate environment exists; board work remains deferred. Continue
Paraformer/HIMLoco, B11 and H8, then complete repository-wide README/behavior
verification and independent review. Do not treat this implementation increment
as completion of the full non-board migration objective.
