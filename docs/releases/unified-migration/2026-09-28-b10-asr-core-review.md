# B10 ASR — Python workflow and portable native core

Date: 2026-09-28. Author implementation record; independent Review=not-run,
Closed=no. This increment does not close ASR or the full migration.

## Source and decisions

The 22 source files were audited against S `380e1a2` in the
[B10 source review](2026-09-28-b10-source-review.md). Published S100/S600
identities remain distinct; S100P/X5 are rejected. Source audio, vocabulary
and historical illustrations are copied unchanged.

The original “CTC” routine omitted repeat collapse. The canonical default
uses CTC (collapse adjacent IDs, then remove blank 0); `--decode-mode legacy`
preserves the old decoding for explicit comparisons. All nonblank tokens
remain literal, and state resets at each independent chunk. This changes
text semantics deliberately, rather than claiming identical source text.

Python retains the source's Fourier resampling and variance-plus-1e-5
normalization before padding. The source native sinc resampling remains a
separate integration requirement. Native normalization now has the same
mathematical contract; double accumulation permits small floating differences.
No claim of full Python/native audio equivalence is made.

## Implementation and customer documentation

- ASR class contains construction and pre_process/forward/post_process/predict
  only. Audio I/O, vocabulary identity, binding, runner, decoding and reporting
  live in separate modules. Transport delegates to the shared raw runner.
- Input is fixed float32 [1,30000]; observed output must be [1,T,3503]. Raw
  FLOAT32 and explicitly described SCALE integer output are checked at load.
  Actual model metadata is not inferred from filenames or fixtures.
- Explicit model downloads, exact external-model identity, scheduling, full-file
  chunk processing, owned output and success/partial-failure reports are present.
  An injected disk-full failure reproduced a secondary exception while writing
  failed.json; the fix preserves both errors on stderr and returns 2.
- Portable C++ CTC/legacy decoding and normalization include invalid input,
  tie, nonfinite and extreme-negative-logit tests. Audio/SDK/task/CLI integration
  is still pending; original native capability is not dropped.
- Shared Unicode edit metrics and the offline evaluator produce micro CER with
  explicit provenance and no silent text normalization. Empty-reference CER is
  null; a scoring result is not proof of authentic model predictions.
- Seven levels of bilingual README cover root, model, Python, C++ core,
  conversion, evaluator and test data. They include concrete parameters, API
  examples, output schemas, decoder differences and failure handling. Historical
  performance illustrations are retained with corrected links and boundaries.
  Conversion lacks an exact checkpoint/export/calibration/compiler recipe; the
  docs list these prerequisites instead of fabricating executable commands.
- Repository/sample indexes and the active S download-script path are updated.

## Evidence

All raw evidence is under [b10-asr-core](evidence/2026-09-28-b10-asr-core/).
`frontend-results.json` contains seven real audio chunks: the bundled recording,
a generated 44.1 kHz stereo recording and a short 8 kHz constant input. Python
prepared tensors equal the unchanged source frontend element-for-element.
Native normalization is compared only after resampling; max differences are
below 2e-6, and `native_resampling_compared` is false.

`readme-results.json` records four executed bilingual examples and 58 checked
local links. The integration example uses real file reading and preprocessing
with a fake SDK; its `AAAAAA` output is a fixture result, not recognized speech.
The CMake README example builds and runs the contract test with ASan/UBSan.
`python-tests.log` records 15 ASR tests. `initial-report-write-failure.log`
retains the disk-full regression before its fix. Shared regressions, publisher
checks and migration contracts are recorded in `host-results.json` and the
corresponding raw logs: 450 related Python tests, 121 publisher tests, and
47 samples with 0 violations, 49 declared skips and 0 exemptions. These checks
do not close the whole migration. Implementation digests are in
`implementation-sha256.json`.

## Still open

Native audio/SDK/CLI integration and its full documentation, Paraformer,
HIMLoco, B11, release/skills/source-tip completeness, whole-branch regression
and fresh independent review remain in the full H0–H9 plan. No board was
contacted. Real SDK/model inference, OE compilation and corpus accuracy
remain not-run; missing source conversion inputs remain explicit limitations.
