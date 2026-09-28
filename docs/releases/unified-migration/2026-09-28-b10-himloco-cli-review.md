# HIMLoco Python binding, offline CLI and model preparation

Base: `2af9c01aaf6d0aae745d52588d94dc0a3befddc5`.
Status: author host implementation; no SDK/board inference, quantization recipe
run or robot control. Native C++, evaluators and source conversion documentation
remain within the unfinished full migration scope.

## Implemented behavior

Exact asset selection binds the X5 publication to the default `model/bayes-e` path.
External paths require the matching asset ID; forged selections fail before SDK
construction. The lazy runner reuses shared local-board checks, publisher SHA-256
verification, metadata reading, scheduling and owned raw arrays. Only one model,
obs_history F32 [1,270] and actions F32 [1,12] are accepted; incompatible output
is rejected, never cast/dequantized silently.

The offline CLI separates model acquisition from execution. Host help/list/preview
need no SDK or network. Execution validates target/hash before SDK loading, reads
source-indexed 1080-byte observations, verifies a colocated manifest when present,
performs the requested warmup, and writes 48-byte action dumps. Model/manifest
hashes are rechecked before success. Reports include environment/module source,
physical metadata, requested scheduling, per-file digests and runner timings.

Each output directory/report must be new. Failures after report reservation retain
completed dumps, current source index, completed warmups and the original error;
partial runs get no aggregate latency summary. Preflight failures write no results.
Timing measures the bound runner, including shared adapter checks/copies, not pure
BPU execution and not exactly the original direct-runtime timing scope.

Explicit download delegates to the shared temporary/hash/atomic installer. This
step executed preview only, not an actual model download. All 21 observation BINs
and their manifest were copied unchanged from the fixed source. Four documentation
levels now have bilingual guides: root, model, Python runtime, test_data. Historical
source performance is retained and distinguished from the synthetic host tests.

## Evidence

- 15 sample tests pass: policy behavior, exact publication/metadata, pre-SDK target
  gate, scheduling through an explicit SDK double, 21-input/warmup/action output,
  failed partial-run reporting, output preservation, duplicate indices and digest
  mismatch. [Tests](evidence/2026-09-28-b10-himloco-cli/tests.log).
- Six actual host CLI cases cover help, list, dry-run, wrong target, host execution
  refusal and download preview. Each leaves the nominated output path absent.
  [CLI records](evidence/2026-09-28-b10-himloco-cli/host-cli.json) include argv/cwd,
  return codes and complete stdout/stderr. The same record confirms 22 source data
  files (21 BINs plus manifest) preserved byte-for-byte.
- Direct sample contract: zero violations, one ordinary CLI policy skip, no
  exemptions. [Contract](evidence/2026-09-28-b10-himloco-cli/contract.json).

This does not establish board runtime compatibility, learned policy action accuracy,
latency or closed-loop behavior. Quantization schemes remain trusted source material
for documentation refactoring under the user's latest instruction; toolchain
availability or rerunning those schemes is not an acceptance blocker.
