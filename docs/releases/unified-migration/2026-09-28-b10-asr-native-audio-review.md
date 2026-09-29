# B10 ASR — native audio and three-stage task

2026-09-28. Implementation record, not independent acceptance. Review=not-run,
Closed=no. Full ASR native SDK/CLI integration and the overall H0–H9 objective
remain open.

## Changes and decisions

`AudioReader` owns the libsndfile handle, validates nonempty audio, reads bounded
interleaved float chunks and reports frame offset/index. Construction failure
and normal destruction release resources. EOF clears the destination; read
errors throw. Numeric preprocessing is separate in `frontend.cc`.

The frontend averages channels, preserves the source's independent-window
`SRC_SINC_BEST_QUALITY` resampling, then uses variance plus 1e-5 normalization
before padding to 30000. Double accumulation avoids float overflow when mixing
channels. Invalid/nonfinite input, oversized chunks and intervals too short for
one target sample are rejected. Valid resampled length comes from the library.

The source's standard-deviation-only native normalization is deliberately
corrected to the Python mathematical contract. Its independent-window
resampling is preserved, rather than claiming continuous streaming. The
[upstream simple API documentation](https://libsndfile.github.io/libsamplerate/api_simple.html)
requires a stateful API for continuous chunked audio. This sample retains
independent inference windows; boundary behavior remains explicitly documented.

`asr.h` exposes construction and pre_process/forward/post_process/predict only.
The task accepts an owned callback, observed positive output-step count and
3503 ordered tokens. Forward validates prepared input and calls transport once;
postprocess validates and decodes owned raw FLOAT32 logits. No file I/O, model
loading or activation is hidden in the task. The SDK adapter remains required
to supply verified model properties and vocabulary identity.

Native FLOAT32 output matches the original C++ capability; Python integer
SCALE support is not claimed for native inference. Tests use synthetic raw
logits and cannot establish actual model metadata or speech accuracy.

## Actual checks

Evidence: [native-audio](evidence/2026-09-28-b10-asr-native-audio/).

- Built actual libsndfile 1.2.2 and libsamplerate 0.2.2 from upstream release
  archives in an isolated prefix. `dependencies.json` records archive hashes,
  configure/build/install commands and exit codes. External codecs are disabled
  in this host libsndfile build; WAV PCM/float is covered, not FLAC support.
- Three C++ tests pass under ASan/UBSan, including an explicit Release build with
  assertions forced on: contract, audio I/O/frontend and three-stage task.
  Tests include 8/16/44.1 kHz, stereo, partial windows, empty/missing files,
  constant/extreme input, invalid geometry, ownership and runner call count.
- Seven audio chunks compare against the actual source read/mix/resample code.
  The comparison extracts that unchanged source prefix and removes only the
  old normalization/padding, then applies the corrected formula separately.
  `audio-results.json` records source/extraction hashes, compiler invocations,
  input/output digests, geometry and differences. This is resampling preservation
  plus intentional normalization correction, not unchanged old final output.
- Python Fourier differences are measured and retained separately, not hidden
  behind a parity assertion. The constant 8 kHz window differs by up to 7.7732
  after normalization because sinc boundary response is not constant, whereas
  Fourier resampling preserves this constant input. This difference also exists
  in the source resampling path; it is not accepted as cross-language parity.
  No Python/native whole-frontend equality is claimed.
- Five bilingual shell/Python examples execute and 60 local links resolve;
  the complete bilingual C++ API fixture compiles and runs with expected `AA`.
  This is synthetic transport output. Both languages document dependencies,
  defaults, stage data, source differences and verification boundaries.
- ASR's 15 Python regression tests pass; full migration contract checks remain
  47 samples, zero violations, 49 policy skips and zero exemptions.

The README evidence checker initially misread the C++ lambda `[](const …)`
as a Markdown link. Its link pass now excludes fenced code; the original
failure remains in `initial-readmes.log`. This was an evidence-harness issue.

Initial missing-interface failures are retained in `initial-audio.log` and
`initial-task.log`. Host library builds do not validate target library ABI or
actual SDK behavior.

## Next required work

Provide the SDK adapter with model/tensor validation, resource ownership and
error handling; integrate target/artifact/vocabulary identity, native CLI and
result recording; execute host fixtures and bring every README current again.
Then continue Paraformer/HIMLoco, B11, H8, full-branch review and remaining H0–H9
work. No board was contacted. SDK/board/model/OE/corpus validation remain not-run.
