# Paraformer native CPU contract and application composition

Date: 2026-09-28. Author implementation/host verification record, not independent
sample acceptance. Base: `927946719906fa35e86dc261c34c5bbae18c34a6`.

## Scope and implementation

The native runtime now has an SDK-independent C++17 numerical library and an
explicit encoder → predictor → CPU CIF → decoder application pipeline. Model
callbacks return owned raw arrays; CIF and text decoding remain separate from
model execution. Invalid features fail before encoder, invalid encoder output
fails before predictor, and zero CIF tokens bypass decoder with absent decoder
timing. File I/O, SDK resources, metadata and identity checks are not hidden in
these numerical interfaces.

The streaming CIF accumulator retains source double cumulative sums rounded to
float32, float32 operation order, padding mask, one emission per frame and the
100-token cap. It is not a generalized multi-fire implementation. Text decoding
retains repeated tokens, first-ID ties, special-token filtering and BPE marker
removal. The old C++ source already handled zero tokens; the Python source's
zero-token failure was fixed separately and is not claimed as a new native fix.

## Verification and reproducibility

Evidence directory: [native-core](evidence/2026-09-28-b10-paraformer-native-core/).

- `red.log` and `pipeline-red.log` preserve missing-interface failures before
  implementation. They are development evidence, not current failures.
- `verify.py` verifies the archived C++ source byte-for-byte against S commit
  `380e1a2bf42041af54be6f34935e50197cfadff9`, extracts its CIF unchanged, compiles
  source and new kernels, and compares 27 deterministic inputs byte-for-byte
  against each other and unified Python. All passed. Twenty native/Python text
  cases using the fixed published vocabulary also passed. See `summary.json`
  for seeds, input/output digests, compiler and environment.
- `check_docs.py` executes the two identical English/Chinese README shell blocks
  from repository root. The Release CMake build with ASan/UBSan passed two CTest
  checks, including assertions enabled in Release. The complete API example
  compiled and returned `2 3 8`. Exact commands and return codes are in
  `doc-summary.json`; current test output is `doc-0.log`, API output `doc-1.log`.
- The two native tests cover hand-derived fractional values, padding, zero/capped
  emissions, invalid inputs, repeat/BPE/special tokens, equal-score ties,
  composition order, zero-token decoder bypass and malformed intermediate data.
- `../rdk_model_zoo/.venv/bin/python -m unittest discover -s samples/speech/paraformer/tests`
  passed 33 tests after the native additions. See `python-tests.log`.

Commands (repository root):

```bash
../rdk_model_zoo/.venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-native-core/verify.py
python3 docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-native-core/check_docs.py
```

These scripts use local compiler/CMake dependencies documented in their source.
The customer README uses ordinary installed CMake/compiler commands and does not
require the migration evidence scripts or coordination directory.

## Documentation and remaining work

Paired native README files provide environment, actual build options, tensor and
ownership contracts, timing boundaries, executable API example and failure
criteria. Root support matrices link this partial library without claiming a
complete native deployment entry.

SDK adapter, prepared-feature/manifest loading, complete native CLI, conversion
recipes and evaluator migration remain open. Real SDK compilation/ABI, HBM
metadata, inference, board tests, OE and dataset CER have not been verified by
this work. Paraformer Refactor remains pending, B10 and H0–H9 remain open. This
record does not replace final independent whole-branch review.
