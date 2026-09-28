# Paraformer real-weight three-stage FP32 export

2026-09-28. Author implementation and host verification, not independent review.
Base `6a7add6b4e7e6db48e60390a4475cb8cad4aba31`. Board/SDK/OE remain not-run.

## Implemented path

`conversion/export.py` strictly loads a local `model.pt` into the pinned
ContextualParaformer configuration and exports the encoder, predictor and decoder
separately. It neither downloads nor overwrites source files. Config, CMVN and
ordered vocabulary must match existing source hashes; model parameters must all
load, with no missing-key fallback or nonfinite values. The checkpoint is read
with Torch's `weights_only=True`.

`torch_stages.py` owns the fixed 400-frame / 100-token stage composition. Predictor
reuses FunASR CNN/tail operations without exporting CIF. Decoder retains the source
padded deployment semantics: count changes its mask, not physical width. It is
adapted from MIT-licensed FunASR; the full notice is included. Helpers, file I/O,
reporting and validation are outside stage forward methods. Upstream export
wrappers mutate only a freshly loaded local model, without global monkey patches.

Ruling: replace source full-graph export and internal-name cutting with explicit
stage exports — existing stage contracts are preserved, while single-feed
constant assumptions and CIF graph surgery are removed. The new path must pass
real-weight source-deployment comparisons, not just shape fixtures. Calibration,
compiler orchestration and dedicated evaluator remain pending.

The graph passes now prove constant ReduceMax/ReduceMin paths, needed for the
encoder's constant length=400 Range. Dynamic Range remains rejected. Output name
binding handles an upstream internal name collision without changing tensor
uses. Output shapes are explicit reshapes in each stage, not unchecked metadata
relabeling. The exporter validates final signatures and full output arrays
against Torch for every supplied check input.

## Actual verification

[Evidence](evidence/2026-09-28-b10-paraformer-export/):

- `download.log`, `hub-files.json`: actual ModelScope download of the official
  source repository. `master` is mutable; observed model hashes are recorded in
  `export-report.json`, not claimed to be an immutable hub revision. Weight files
  and generated ONNX files are local ignored artifacts, not added to Git.
- `weight-load-probe.log`: every weight key matched. Config/CMVN/vocabulary hashes
  match the source; initial source/new Torch decoder check at width 100 has zero
  difference.
- `export-v1.log` / `export-v1-failed.json`: symbolic output geometry rejected.
  `export-v2.*`: internal tensor-name collision rejected. `export-v3.*`: constant
  ReduceMax not supported by the proof whitelist yet, so Range rejected. Those
  failures led to explicit reshapes, safe physical output rebinding and the
  narrowly extended constant whitelist, respectively. None was bypassed by
  claiming success or overwriting the failed run.
- `export-v4.log`, `export-report.json`: all three real-weight stages exported.
  Four encoder checks, four predictor checks and eight decoder checks passed
  (`rtol=1e-4, atol=1e-4`). Inputs include zero/random features and both actual
  prepared source audio arrays; decoder additionally checks counts 0/1/17/100.
  Maximum absolute differences: encoder 3.4422e-6, predictor 2.9803e-7, decoder
  4.8638e-5. Final names/dtypes/shapes match the existing S100 stage contract.
- `red.log`: stage adapter tests initially failed because the implementation was
  absent. `all-tests.log`: **56 sample tests passed with no skips** in the real
  Torch/FunASR/ONNX/ORT Python 3.12 environment. They include ten graph tests, stage
  CNN/tail/mask checks, output-name collision and exporter preflight/comparison
  failures. `main-env-tests.log` is an earlier 54-test run with two documented
  optional Torch skips, not the final no-skip result.
- `verify_real.py`, `real-comparison*.json`/`.log`: two complete actual-feature
  pipelines have identical token counts, all token IDs and text between Torch
  and ONNX Runtime. Both ORT default optimization and disabled optimization were
  exercised. The reference transcripts are not reproduced perfectly, so these
  are inference equivalence examples, not CER/accuracy acceptance.
- The same verifier exports the unmodified upstream decoder then executes the
  archived `05_fold_range.py`; only its fixed temporary probe path is redirected
  to a fresh directory. Source code hash is recorded. This reproduces the
  historical fixed-100 deployment transformation without touching source files.
  Old/new ONNX full output arrays are **exactly equal** for counts 0/1/17/100 under
  each matched ORT setting. Source/new Torch at count 100 is also exactly equal.
- `docs-check.json`: bilingual API examples and local links. Explicit downloads,
  environment installation and full export are evidenced by the separate actual
  run logs, not silently triggered by the documentation verifier.
- `migration-gate.json`/`.log`: 47 samples, zero violations, 49 documented policy
  skips, zero exemptions. Paraformer remains pending, not promoted by this gate.

Reproduce with the dependency set in `conversion/requirements-export.txt`:

```bash
python samples/speech/paraformer/conversion/export.py --model-dir /path/to/local/source --output-dir /path/to/new/export --feature /path/to/prepared.npy
python -m unittest discover -s samples/speech/paraformer/tests -v
python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-export/verify_real.py --model-dir /path/to/local/source --export-dir /path/to/export
```

The comparison script uses the two committed prepared-feature examples, verifies
their hashes and loads real source weights. Its optional
`--disable-optimizations` selects the second ORT comparison mode.

## Numeric limitations retained, not waived

`unpadded-reference-failed.log` records that comparing a shorter unpadded upstream
sequence to the fixed-width deployment is not equivalent. This was an incorrect
reference boundary for the deployed graph; the fixed-100 source transform above
is the applicable migration reference. No claim of original variable-length
model equivalence is made.

`padded-reference-failed.log` and `padded-reference-noopt-failed.log` retain failed
Torch-versus-ORT stress comparisons using arbitrary random context/acoustic
vectors. Default ORT differences reach 1.7572 in that separate stress set;
disabled optimization reaches 3.0284. These do not satisfy the export tolerance.
The matching old/new ONNX arrays remain exactly equal, so the migration preserves
the old exported behavior on these cases; this does not resolve or establish a
cause for the upstream cross-framework numerical difference. The 16 passing
export cases use contexts from the actual encoder, and cannot be extrapolated to
all possible shape-valid hidden inputs. The README pair discloses both facts.

The example transcripts are `广州市房地产中介协会析` and
`新地网的诞生迅速绞热南沙土地市`; they contain errors against the supplied reference
texts. No dataset score, accuracy gain, quantized-model or board result is claimed.

## Remaining work

Unified calibration, nash-e compilation preparation/execution boundaries and
specialized evaluator/documentation still need migration. Updated sample and
conversion README pairs show the implemented export path and those gaps; they do
not point customers to a pretend end-to-end compilation command. Refactor/Docs
remain pending, Review=not-run, Closed=no. B10 and H0–H9 remain open.
