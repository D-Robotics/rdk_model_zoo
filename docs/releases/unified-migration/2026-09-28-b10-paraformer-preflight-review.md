# Paraformer production group preflight

Date: 2026-09-28. Author implementation and host checks, not independent sample
acceptance. Base `db2e038ff037623d77e8ce755ce77b3aaa9cd4b8`.

## Result

The SDK-independent `paraformer_preflight` library validates a complete three-model
S100 group and fixed published vocabulary. `make_preflight` uses the real shared
local-identity reader, immediately validates all three models before any runner
construction, and returns the mandatory SDK callback. The callback rechecks exact
selected stage/target/path, current identity and all model/vocabulary digests at
each load. CIF, text decoding and SDK inference responsibilities remain separate.

Checks include exactly one encoder/predictor/decoder, matching published asset IDs,
S100 only (S100P board aliases rejected), 64-digit case-insensitive expected SHA,
regular nonempty model files, distinct canonical files and hard-link identities,
model content digests and fixed 8,404-token vocabulary digest. Expected model
digests are caller-observed file identity, not publisher authentication: current
publication metadata has no authoritative model digests. Runtime tensor metadata
checks remain necessary. Files must remain unchanged during use; preflight is
not a filesystem lock against concurrent replacement.

## Evidence

[Evidence directory](evidence/2026-09-28-b10-paraformer-preflight/).

- `red.log`: absent implementation before adding the preflight code.
- `configure.log`, `build.log`, `ctest.log`: Release build under ASan/UBSan; four
  native tests pass. New preflight tests use an explicitly test-only identity
  reader and actual files/digests/vocabulary. They verify arbitrary group order,
  wrong stage/asset, S100P/unknown/other identity, wrong digest, missing/empty/file
  alias, bad vocabulary, changed model and callback model mismatch. A decoder
  mutation is rejected even when invoking the encoder's preflight callback.
- `doc-0.log`, `doc-1.log`, `api-compile.log`, `doc-summary.json`: both bilingual
  shell examples run, four tests pass, numerical output remains `2 3 8`, and the
  updated SDK embedding function compiles using the concrete preflight factory.
- `vendor-sdk-configure.log`: missing real SDK headers rejected. Host checks do
  not establish vendor ABI or HBM metadata compatibility.
- `local_probe.cc` and `local-probe.log`: linked against the production preflight
  library and actual local identity reader (no identity double), this Mac rejects
  factory creation with `Paraformer requires actual local S100 identity`, before
  examining model/vocabulary paths. This is a negative host check, not a board run.

Reproduce documentation checks from repository root:

```bash
python3 docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-preflight/check_docs.py
```

Production identity negative probe after the recorded build:

```bash
c++ -std=c++17 -fsanitize=address,undefined -Isamples/speech/paraformer/runtime/cpp/inc -Isamples/_shared/cpp docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-preflight/local_probe.cc ../.coordination/paraformer-native-core/libparaformer_preflight.a -o /tmp/paraformer-local-preflight
/tmp/paraformer-local-preflight
```

The probe deliberately requires an unrecognized host; do not interpret its result
on a supported board as a deployment test. Migration checks are recorded separately
in `migration-gate.json`; Paraformer still has incomplete migration work.

## Documentation and open work

English/Chinese native documentation now includes the full model-group API,
publication IDs, vocabulary digest, actual identity precedence, fail conditions,
hashing cost, ownership and provenance limits. The embedding example now creates
its own production preflight callback. Root sample status and the plan/ledger are
updated without declaring a complete native command.

Prepared-manifest loading, native CLI/results, conversion/evaluator migration and
full sample acceptance remain pending. Real SDK/board/OE/model execution remain
unverified. Continue Paraformer, HIMLoco, B11/H8 and H0–H9 without narrowing the
original full non-board objective or closing final independent review.
