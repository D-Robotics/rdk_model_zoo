# Paraformer complete native application and launcher

Date: 2026-09-28. Author host implementation/verification, not independent sample
acceptance. Base `44e3c6075e3202a8c14192f2980735cb72efcc01`.

## Result and architecture

`paraformer_demo` now connects production model-group preflight, fixed vocabulary
loading, prepared-manifest/NPY I/O, three independent raw SDK runners, the existing
CIF/text application pipeline and atomic success/failure reports. The model runner
still performs one raw call; file parsing, command-line handling, report writing
and text/CIF remain outside its inference method. Native input remains the actual
Python FunASR prepared features, preserving the source's Python → C++ capability.

The public `run.sh` delegates to `launcher.py`: host model listing/preview,
publication resolution, explicit model overrides, actual local identity gate,
optional real SDK build, complete process logs and validation of native results.
No auto-download/install, remote/board connection or fake-backend fallback exists.

The native CLI requires exact stage paths/asset IDs/digests, target, prepared
manifest, vocabulary and new output directory. The launcher fills those arguments
from active manifests and observed files. Production preflight checks all three
artifacts before any SDK load. Native reports retain all three selected artifacts,
runtime physical names/roles/types/shapes/strides, input identities, source
annotations, recognition IDs/text, frame/truncation state and stage timing.

The launcher accepts success only after rc=0, a regular result file, native-sdk
backend, exact artifact/input identity, Python-binding-compatible tensor metadata,
valid token/text/timing/frame semantics and final file rehashes. A binary marked
host-fixture is rejected at its help check before inference. These checks establish
report consistency; they do not replace real SDK/board validation.

## Failure semantics

Both layers refuse existing output directories. Pre-directory parser/preflight
failures produce stderr/rc=2 without a new run directory. Later native failures
write `failed.json`, preserving completed records, current utterance and error;
successful result files are not emitted. The launcher retains failed build/run
logs and its failed launch report. Report-writing failures are themselves printed.

`inference_attempted` becomes true before a pipeline call. `inference_executed`
is false before calls, null after a failed first call (partial model execution
may have happened), and true after any complete utterance. Zero CIF tokens bypass
decoder, return empty text/IDs and leave decoder timing null. Stage timing excludes
frontend/I/O and does not establish BPU performance or dataset CER.

## Host evidence

[Evidence directory](evidence/2026-09-28-b10-paraformer-native-cli/).

- `configure.log`, `build.log`, `ctest.log`: actual Release+ASan/UBSan build; six
  native tests passed. `paraformer_cli_fixture` uses explicit test-only model
  transport and identity doubles and declares backend `host-fixture`. It is not
  linked into or installed as `paraformer_demo`.
- `verify.py`, `summary.json`, `cli-checks.log`: 14 complete native application
  subprocess cases passed, using real persisted frontend arrays and fixed
  vocabulary. They cover help/invalid arguments, two inputs, selected prefix,
  output reuse, bad decoder digest before output, empty-CIF bypass, encoder/decoder
  failure, model-load failure and a second input failure retaining first progress.
  All report payloads retain the host-fixture marker. The synthetic `andand`
  transcript is not ASR accuracy evidence.
- `python-tests.log`: 39 Paraformer tests passed (33 existing + six native launch/
  report tests). New checks cover valid/zero-token reports, malformed or forged
  identity/backend/metadata/token/timing/frame reports, preview/errors, local gate
  ordering and refusing a marked fixture binary before inference.
- `check_docs.py`, `doc-summary.json`, `doc-*.log`, `api-*.log`: five distinct host
  README shell blocks ran and both complete C++ API examples compiled. The actual
  FunASR preprocessing block ran in a fresh root-layout directory with `samples`
  pointing to this checkout; resulting feature bytes match the previous real
  frontend evidence exactly. Four base tests and six optional I/O/CLI tests passed.
  The download/board block is explicitly not-run.
- `build-gates.json`, `cli-without-sdk.log`, `real-sdk-required.log`: CMake rejects
  incomplete CLI configuration and missing real SDK dependencies. No vendor ABI
  success is inferred from API-double compilation.
- `local-launch-reject.log`: actual Mac launcher rejects unknown local board
  identity before output creation, without an identity double or model loading.
- `migration-gate.json`: 47 samples / 0 violations / 49 documented policy skips /
  0 exemptions. Paraformer is still not promoted to completed migration scope.

Reproduce host application checks after building the I/O/test targets:

```bash
../rdk_model_zoo/.venv/bin/python docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-native-cli/verify.py
python3 docs/releases/unified-migration/evidence/2026-09-28-b10-paraformer-native-cli/check_docs.py
../rdk_model_zoo/.venv/bin/python -m unittest discover -s samples/speech/paraformer/tests
```

The recorded coordination build path is explicit in `verify.py`; the customer
README uses ordinary CMake and launcher commands. Native CLI checks use real
application code with substituted test transport, never a simulated vendor claim.

## Documentation and remaining scope

Root, Python and native English/Chinese READMEs now agree that both runtime entries
are implemented and host-checked, while actual SDK/model/board execution is not
verified. The native guide starts with preview → explicit frontend preparation →
board build/run, documents all launcher/direct flags, dependencies, output layout,
partial-failure meaning and host-versus-board boundaries. Library APIs remain
available below the runnable flow.

Paraformer conversion/evaluator migration and complete sample acceptance remain
open. No real vendor SDK ABI, HBM inference, board, OE or dataset CER was run.
HIMLoco, B11/H8, full README audit, integration/regression and final independent
whole-branch review continue under the original full non-board objective.
B10 and H0–H9 remain open; this implementation is not a smaller completion goal.
