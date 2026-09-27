# YOLOE native CLI and customer workflow

Base: `010b840a781b5c10b3320afcdd81d28571aeee44`. Author implementation/verification record;
independent review remains not-run. Board/real SDK/OE/dataset acceptance remains
not-run. B9/H5 and the full H0–H9 goal remain open.

## Implementation

The canonical launcher resolves exact publication identities through the Python
binding table, gates real local identity and file bytes before explicit builds,
and runs the native executable with a verified expected model digest. S published
quantized models are refused; custom float paths require their expected SHA-256.
No implicit download, identity override or target fallback is introduced.

The native entry performs preflight before image/output/SDK work, uses the real
three-stage task, and writes an annotated image, binary ROI PNG masks and a final
identity-bound JSON report. Empty ROI axes are preserved without invalid PNGs.
CLI parsing and all file/rendering work live outside the inference task.

The launcher retains exact argv/cwd, UTC times, raw stdout/stderr, return codes
and model/input/vocabulary/binary/result hashes. Success without a matching
native-sdk report is rejected, including host-fixture output. Build failures do
not continue into inference. Existing outputs are never overwritten.

Shared Python configuration moved to a module without image or SDK imports;
existing Config and validate_config import locations remain compatible. SDK
adapter and CLI now share the same allowed target/variant predicate.

## Customer documentation

Both native READMEs cover capabilities and model availability separately, explicit
download/build/custom conversion paths, every launcher option, exact output and
failure interpretation, reusable stage/SDK APIs, and host-test commands. Existing
algorithm contracts/source evidence are retained, including native ROI versus
X5 Python full-image probability masks and E11 NMS equality differences. Root
sample matrices and navigation now reflect the implemented entry and unchanged
SDK/board validation limits.

## Evidence

Evidence directory: [native CLI](evidence/2026-09-28-yoloe-native-cli/).
The explicit fixture uses actual main/options/preflight/image/result code and real
OpenCV, but replaces board identity and SDK execution with a synthetic S100P
backend. Its report declares `host-fixture`; no fixture run is SDK or board proof.
The input, non-deployable model fixture, outputs, hashes, commands and logs are
retained in `fixture-runs/20260927T232538281496Z/`. Eight commands behaved as expected:
help, positive E26, wrong target, wrong model digest, wrong vocabulary, illegal
E26 NMS, duplicate option, and output reuse. Result checks include exact class,
label, sigmoid score tolerance, 4×4 binary ROI with nine foreground pixels,
input/model identity, 64×64 annotation and no overwrite.

Initial failures remain in evidence: missing native CLI header during test-first
implementation; raw non-UTF-8 log handling and dangling output symlink handling;
and a JSON array report causing an uncaught AttributeError. Fixes retain raw
bytes, reject a pre-resolution output symlink and validate JSON object type before
field access. The latter now produces a failed launch record with complete logs.

Final `run_host.py` results are recorded in [host-results.json](evidence/2026-09-28-yoloe-native-cli/host-results.json):

- 557 Python checks: YOLOE 44 (including 13 launcher tests), Ultralytics 143,
  shared 153, ResNet 52, OCR 44, checker 27, export 5, evaluator 10, YOLOv5 79.
- Both bilingual documented CMake workflows build and pass all 11 YOLOE native
  tests with ASan/UBSan and actual OpenCV 4.14.0. Shared native regression: 12/12.
- Both reusable API examples compile. All bilingual host shell blocks match
  and were executed; board shell examples were deliberately not executed.
- Missing real SDK dependencies produce the expected configure failure with
  `YOLOE_BUILD_CLI=ON`; this is a rejection check, not a real SDK build.
- Migration contract checker: 45 samples, zero violations, 47 declared skips,
  zero exemptions. README API examples and local links pass.
- [route-results.json](evidence/2026-09-28-yoloe-native-cli/route-results.json)
  retains 18 actual host CLI checks: list, all 14 exact publication selections,
  S600/auto-dry-run rejection and published S execution refusal. Separate import
  isolation verifies the launcher needs neither Python OpenCV nor hbm_runtime.

These results do not certify vendor SDK ABI or board compatibility. The actual
fixture uses synthetic tensors, and launch policy tests use explicit process
mocks; neither is relabeled as a measured model inference result.
