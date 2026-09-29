# HIMLoco independent host runtime review

Reviewer: Codex. Base `d4bb9ade`. Accept the inspected Python/C++ offline runtime
migration and corresponding usage/contract documentation within host scope. No
blocking finding identified here; the reviewer changed no implementation. This
does not certify robot control, board behavior, conversion or all B10 samples.

Reviewed the Python task, exact binding, input/provenance loading and application
reporting; native policy, SDK ownership, metadata/storage validation and model
preflight; root and evaluator guides and the recorded runtime contracts. The
model boundary consumes 270 prepared observations and returns 12 raw float32
actions. Stages do not resample/reorder history, apply 0.25 action scale, update
history or send robot commands. Outputs and per-call timing are owned separately;
Python measures its bound runner and native timing surrounds infer/wait. Neither
is silently presented as the same measurement scope.

Actual execution requires X5 identity and hash-pinned publication before SDK
construction. Native compact input submission and aligned output traversal retain
the source convention; fake SDK tests cover that convention and cannot certify
all vendor layouts or ABI versions. Packed model, allocations and tasks have
scoped cleanup, with injected error paths. Input files retain numerical source
indices, byte order and hashes; present manifests bind expected files and source
rows. Runtime reports distinguish completion from partial failures and refuse
existing output/report paths rather than overwriting them.

The guide preserves current-first six-frame observations, controller scaling as
an external responsibility, 21 bundled observations, historical environment,
latency distributions and differing timing scopes. It distinguishes independently
trained checkpoints, model output agreement and actual control-loop behavior.
Conversion and model evaluation recipes remain inherited source material under
the user's trust decision; no export/calibration/quantization evaluation was run
or required to accept this bounded runtime package.

## Independent verification

[verification.json](evidence/2026-09-28-himloco-independent-review/verification.json)
records complete commands/output and candidate hashes. All recorded hashes were
rechecked before this report. The 23-test sample suite passes and the checker
reports zero violations, one CLI policy skip, zero exemptions. Tests include
Python stages, exact metadata/identity, file/provenance/report behavior, native
compiled policy, CLI, preflight and SDK doubles. No board, downloaded model,
robot controller, toolchain installation or vendor-linked executable was used.
Real vendor ABI, runtime accuracy/performance and closed-loop behavior remain
not-run. H6/B10 and full integration/README acceptance retain separate open scope.
