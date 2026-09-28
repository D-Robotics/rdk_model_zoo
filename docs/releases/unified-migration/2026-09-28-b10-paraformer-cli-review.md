# B10 Paraformer Python CLI and feature export — host verified

Status: full Python command implemented; real host preprocessing verified. Actual
S100 HBM/SDK execution remains not-run. Native C++, conversion/calibration,
evaluator and full sample/branch acceptance remain open.

## Behavior and source repairs

The CLI provides help/list/dry-run without frontend/SDK initialization; explicit
frontend-only preparation; and board-gated three-model inference. It supports the
source manifest/audio-dir and per-model path options, adds exact asset IDs, single
WAV input, scheduling, fixed seed and fresh result directories. External model
paths require all three declared identities. No pip, downloads, virtualenv creation
or legacy positional data-dir dispatch is hidden in run.sh.

Source preprocess-only rewrote the caller's manifest, and inference silently
skipped missing files. The unified application validates unique filename-stem IDs
and string references, fails on missing selected audio, and writes a separate
prepared-manifest.json preserving input annotations plus newly computed feature
metadata. Output NPYs are named under feats/; source manifest and audio are never
written. Existing output directories are rejected. Positive max-utts selects a
prefix after structural manifest validation; zero means all.

The application keeps I/O/reporting outside the numerical frontend and raw model
runners. Success reports capture identities, observed hashes, bound metadata on
inference, truncation, separate reference/prediction text, stage times and decoder
bypass. Preprocessing produces no transcript or invented inference timings.
Failed runs retain current/previous utterance state when the output is writable;
earlier preflight errors may have no output directory. Byte checks before success
cover all used audio and configuration/model files. Inputs that change are rejected.

`inference_attempted` means pipeline entry; `inference_executed` is false for
preprocessing, true after a successful pipeline return, or null after a first
failed attempt whose execution completion cannot be asserted. A prior successful
utterance keeps it true on a later failure. None of these fields certifies board
validation. Model metadata and token order remain separately validated.

## Evidence

Seven new CLI tests join the existing 26, for 33 sample tests. They cover manifest
and directory protection, explicit errors, no-load preview, target gate before
frontend/output work, partial-failure reporting and the complete inference CLI
using **explicit SDK doubles** with real shared runners and the actual pipeline.
The synthetic `andand` result from that unit fixture is not a model transcript.

[Real CLI records](evidence/2026-09-28-b10-paraformer-cli/real-cli-summary.json)
contain ten actual commands with precise argv/cwd, UTC bounds, return codes and
full stdout/stderr. Real FunASR processes both bundled WAVs into valid lengths
71/78; the records retain NPY files, hashes, result JSON and separate manifests.
Additional commands cover single audio, limit=1, existing output rejection,
unsupported target, invalid rate with failed.json, and real host inference refusal
before creating an output directory. The input manifest is byte-unchanged.
No HBM model was downloaded or executed for those CLI runs.

Reproduce in the documented real frontend environment, from repository root:

```bash
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-cli/verify_real_cli.py
```

Every rerun creates a fresh run directory under the evidence location. The previous
raw records are not replaced. Unit/related shared regression, migration scope and
document command results are recorded alongside those runs.

## README quality and remaining scope

The new bilingual sample root includes the full model-preparation → inference
command path, an independently runnable board-free path, support and validation
matrix, exact observed frontend results, dependencies, file layout and licensing
boundaries. Runtime docs enumerate all CLI arguments/defaults, result schemas,
reference-versus-prediction meaning, failure behavior, source incompatibilities,
truncation and timing limits. Existing complete CPU examples remain available;
board API snippets now define their feature inputs rather than using undefined
variables. The identical bilingual API snippet was executed with the real FunASR
frontend and explicitly synthetic SDK outputs; all three model calls and scheduling
delegations were observed. See [API fixture evidence](evidence/2026-09-28-b10-paraformer-cli/api-example.json);
its `andand` text is not a model prediction. Model/test-data docs are synchronized. Original evidence scripts only
replay their original blocks, not newly appended install or inference commands.

Remaining: native C++ consumer/launcher, all conversion scripts with shared CIF,
complete evaluator and historical benchmark attribution, final README normalization
and independent whole-branch review. No H0–H9 or B10 closure is asserted.
