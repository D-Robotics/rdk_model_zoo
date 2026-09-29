# Paraformer native prepared-feature reader

Date: 2026-09-28. Author implementation/host checks, not independent acceptance.
Base `4c7e23a26f289caaf0f537af4fee09c4c95ff95b`.

## Implementation

The optional `paraformer_feature_io` library reads the actual prepared JSON manifest
and NPY files emitted by unified Python `--preprocess-only`. Audio preprocessing
remains the real FunASR Python frontend, preserving the source Python → C++ bridge;
there is no alternative filterbank implementation. File parsing remains separate
from model inference, CIF and text decoding.

Manifest parsing requires unique IDs and explicit valid/original frame counts,
truncation state, feature path/digest and optional reference text. Unknown
annotations survive semantic JSON reserialization. Structure is validated for all
records before optional prefix selection. Relative paths become absolute paths
against the manifest directory, independent of subsequent cwd changes. Nonselected
feature files need not exist, but their manifest records must still be valid.

The loader reads one bounded owned byte buffer, verifies its SHA-256, parses NPY
as data without evaluating expressions, and returns native float32 values. It
supports versions 1.0/2.0/3.0, little/big-endian float32 and fixed C-order shape
[1,400,560]. Header key order/quote style are independent. Malformed metadata,
nonfinite values, wrong geometry/type/order, unsupported versions, duplicate keys,
trailing syntax/data, missing/partial files and digest mismatch are rejected.
Header length is bounded at 64 KiB. The digest binds parsed bytes, not publisher or
frontend provenance.

## Verification

[Evidence](evidence/2026-09-28-b10-paraformer-feature-io/):

- `red.log` preserves the absent implementation before coding.
- `configure.log`, `build.log`, `ctest.log`: Release+ASan/UBSan native build with
  the real nlohmann JSON 3.11.3 headers; all five tests passed.
- `verify.py`, `summary.json`, `parity.log`: 32 subprocess cases passed against
  the production reader. The two persisted real FunASR WAV feature arrays have
  exactly the same native value bytes as NumPy, with 71/78 valid frames and
  original reference annotations retained. Synthetic cases include all supported
  NPY versions, endian conversion, 500→400 truncation, case-insensitive digests,
  selection semantics, reordered/double-quoted header and negative format,
  metadata, digest, finite-value and partial-file cases. Wrong-file digests are
  recomputed deliberately for malformed-format cases so they reach the parser.
- `check_docs.py`, `doc-summary.json`, `doc-*.log`, `api-*.log`: all three distinct
  bilingual shell examples executed. Base build passed four tests; optional I/O
  build passed five. Numerical example printed `2 3 8`. Both complete C++ API
  functions compiled with warnings treated as errors. The custom JSON prefix is
  recorded explicitly; no dependency was silently installed.

Reproduction from repository root, after the documented I/O build (or use the
recorded coordination build configured in `verify.py`):

```bash
../rdk_model_zoo/.venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-feature-io/verify.py /tmp/rdk-paraformer-io/feature_probe
python3 docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-feature-io/check_docs.py
```

`feature_probe` is explicitly a host file-reader tool; it never loads an SDK or
runs inference. The archive's original C++ reader was examined to preserve the
feature bridge, but acceptance here compares actual NumPy values rather than
assuming the source parser's incomplete-read behavior is correct.

## Documentation and remaining work

Both native README files now cover JSON dependency/build switches, actual reader
API, all manifest fields, optional selection/reference semantics, fixed geometry,
endianness, resource limits and failure criteria. Existing numerical-only build
continues to work without JSON. Root status, execution plan and ledger record
this progress without promoting the sample to complete.

Full native CLI, model pipeline wiring, success/failure result files, conversion
recipes and evaluator remain open. No SDK ABI, HBM inference, board, OE or dataset
CER was validated by these checks. Continue the entire B10/B11/H8/H0–H9 objective;
final whole-branch independent review remains pending.
