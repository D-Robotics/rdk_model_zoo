# Paraformer real calibration and explicit compilation preparation

2026-09-28. Author implementation/host verification, not independent acceptance.
Base `f7a8d69e49ebd1cad387e59ec81e0856a7d15140`.

## Implemented workflow

`conversion/prepare.py` validates a completed three-stage export and actual ONNX
signatures, snapshots models/CMVN/export report into a new workspace, then uses
the shared real FunASR frontend on a sorted prefix of selected 16 kHz WAVs.
It does not construct a full AutoModel, create random calibration inputs, resample
unsupported audio or silently skip a malformed selected item. It records actual
frame counts/truncation, CPU seed, dependency versions and input identities.

`onnx_stage.py` performs one named CPU ONNX call; calibration orchestration and
file I/O remain separate. Encoder/predictor outputs feed the existing CPU CIF
with explicit `real_T=None`, preserving the source unmasked calibration behavior.
All seven aligned NPY streams have fixed shape/dtype/finiteness contracts. Shared
CIF handles a zero-token result without dropping the sample. Runtime inference
continues to apply the utterance's valid-frame mask.

Three generated nash-e YAML configs preserve the source non-path settings and
prefixes. Paths use one consistent workspace root, replacing the archived
out-directory/config-relative-path contradiction. The source/default sample
count is 50, but fewer files remain visible as a warning and an actual count;
this is not calibration quality or dataset coverage acceptance.

`compile.py` is an explicit, separate operation into a new run directory. It
validates all snapshot/config/calibration hashes, rejects added NPY inputs and
checks actual calibration shapes/dtypes/finite values. Per-run configs remap paths
without altering the prepared workspace. Exact argv/cwd, config hash, compiler
file hash, UTC times, complete separate stdout/stderr, return codes and artifact
identities are retained. Process-start failure, nonzero exit, and zero exit with
no nonempty expected HBM stop later stages. Final preparation identity is checked
again. Only `compiled_unverified` is possible on nominal compiler success; no
SDK, board, accuracy, publisher identity or final I/O-precision acceptance follows.

## Actual evidence

[Evidence directory](evidence/2026-09-28-b10-paraformer-calibration/):

- `red.log`: calibration module absent before implementation.
- `calibration-tests.log`: first test assertion compared macOS `/var` to its
  resolved `/private/var` alias; the expectation was corrected to the specified
  canonical-path result, without weakening selection checks.
- `spawn-failure-red.log`: a compiler process that could not start left a stage
  incorrectly marked running. The fix records failed/null return code/end time;
  the full suite includes this regression.
- `all-tests.log`: **65 sample tests passed without skips** in the complete
  Torch/FunASR/ONNX/ORT environment. New coverage includes deterministic selection,
  nonempty/positive constraints, exact unmasked CIF reuse, invalid arrays,
  source-recipe parity, successful-but-unverified orchestration, retained logs,
  partial failure, missing artifacts, process-start failure, altered/extra
  calibration inputs, missing compiler and output reuse. Compiler subprocesses
  in these unit tests explicitly emit fixture bytes; they are not vendor proof.
- `prepare-real.log`, `prepare-real-v2.log`, `preparation.json`: actual real-weight
  FP32 encoder/predictor and real frontend processed both committed WAVs. The
  second run adds explicit seed/environment reporting; no failed run is hidden.
  Fourteen NPY arrays were generated (seven roles × two utterances), and three
  configs were prepared. Source/model/data hashes and value ranges are recorded.
- `source-calibration.log`: the unchanged archived `10_gen_real_calib.py` actually
  ran on the same exported FP32 models and prepared speech features.
  `verify.py`, `verify.log`, `summary.json`: all twelve derived arrays are byte
  equal to its outputs; the two speech arrays are byte equal to the previously
  recorded unified real frontend results. This checks shared calibration math,
  not the old script's unspecified/global frontend random-state behavior.
- `check_failure.py`, `wrong-rate.json`/`.log`: actual CLI run with a selected
  8 kHz WAV returned 2, retained `preparation_failed` and current input, recorded
  zero completed samples, and produced no compiler configs. It did not report a
  skipped-file success.
- `summary.json`: this host has neither `hb_compile` nor Docker. The actual
  compile entry returned 2 on missing OE before creating output. Real OE/SDK/
  board and quantized accuracy are **not-run**.
- `encoder.yaml`, `predictor.yaml`, `decoder.yaml`: actual generated configs.
  They preserve max/INT16, O2 latency, single core, 32 jobs, cache disabled and
  featuremap/NCHW source settings; paths are relative to a complete workspace.
- `docs-check.json`/`.log`: current bilingual links, pure graph API examples and
  actual CLI help. Full preparation and missing-compiler commands have separate
  real evidence above; the checker does not silently download or compile.
- `migration-gate.json`/`.log`: current migration checker result. Paraformer is
  still pending until its evaluator/full sample documentation are finished.

The two WAVs are workflow verification only; 50 representative calibration
utterances and dataset quality have not been established. Published HBM assets
are untouched. The earlier export stress-test numerical limitations remain
recorded in the export report and customer README.

## Reproduction and remaining work

Use the export environment documented in the conversion README, then:

```bash
python samples/speech/paraformer/conversion/prepare.py --export-dir /path/to/export --wav-dir samples/speech/paraformer/test_data --sample-count 2 --output-dir /path/to/new/preparation
python -m unittest discover -s samples/speech/paraformer/tests -v
```

The README pair now covers export → calibration/config preparation → explicit
compilation, every option, all tensor streams, artifact paths, report states,
retry/failure rules and real toolchain prerequisites. Root and Python guides are
synchronized with that implementation boundary. The source's bare image tag is
preserved as historical context, without inventing a registry or availability.

The dedicated evaluator and its complete bilingual guide are next. Paraformer
Refactor/Docs remain pending, independent Review=not-run, Closed=no. Actual OE
and board validation remain separate external-environment gaps. All H0–H9 work
continues; this increment does not close the full migration.
