# Shared single-array runner consolidation

Status: implemented and host-tested. This is an H8 consolidation subtask and
preparation for the remaining B8 samples, not closure of B8 or H8.

## Change and boundary

PointNet and UNet had almost identical SDK loading, identity/artifact gates,
scheduling, container validation and raw-output ownership. These now live in
`samples/_shared/single_array_runner.py`. The two sample modules are small
adapters declaring their binding loader and physical input shape/dtype.

Ruling: share only the single-input/single-output array transport supported by
these two real consumers. Preserve sample-specific artifact selection, logical
metadata, input preparation and output interpretation. This does not create a
universal inference base class or migrate multi-stage/stateful consumers implicitly.

The constructor and public methods of both sample runners are unchanged. Real
loading still verifies target identity and artifact before SDK construction;
explicit injected runtimes remain documented test/evaluator seams. Binding
failure discards runtime state; scheduling with no values does not load the SDK.
Raw output remains an owned array and receives no numerical transformation.

## Host evidence

Initial shared-module tests fail before implementation. Six transport tests now
cover differing physical/logical input shape, output ownership, invalid inputs
not reaching SDK, malformed output rejection, failed-binding cleanup, scheduler
semantics and identity/artifact checks before SDK. They are included in the 137
shared tests. PointNet 21 and UNet 20 tests pass unchanged: 178 total.

The sample suites exercise the actual README examples and CLI output writing with
injected SDK fixtures, as well as source preprocessing/postprocessing parity.
Contract scope remains 38 samples / 0 violations / 39 policy skips / 0 exemptions.
No board test, download or new independent acceptance is claimed.

[Evidence/logs](evidence/2026-09-26-array-runner-evidence.json).

## PP-LiteSeg findings driving the next implementation

Seven source files were compared byte-for-byte with X5 source pin
`ac115717197920355fc390bb04299b20e6436864`. Findings:

1. Root/conversion READMEs describe logits followed by argmax, but runtime treats
   output as an int32 class map and only squeezes/casts. The migration must not
   add a second argmax. Actual published-BIN metadata remains board-unobserved.
2. The default `street.jpg` does not exist; source test data contain `street.png`
   and `test.jpg`. Fix and test the default path rather than carrying the typo.
3. Runtime imports the SDK eagerly and has a no-op scheduling setter. The
   canonical entry must stay host-safe and not claim a silent scheduling success.
4. Export does not require a checkpoint and depends on unpinned external
   PaddleSeg configuration. Require explicit checkpoint provenance and inspect
   graph output semantics before claiming compatibility with the runtime.
5. `build_bin.sh` can exit successfully after failing to find its expected BIN.
   Preserve build capability while making missing-output status explicit failure.
6. Calibration uses basename stems and reuses output directories, risking silent
   overwrites for nested inputs or reruns. Preserve numeric format while fixing
   run ownership and deterministic naming.

[Source audit and hashes](evidence/2026-09-26-b8-ppliteseg-audit.json).
These are findings, not completed PP-LiteSeg fixes. No canonical PP-LiteSeg sample
has been added yet. Continue its complete code/docs/conversion migration next,
then the other five B8 families and full H0–H9 scope.
