# HIMLoco native application, build and launcher

Base: `ed7dcb0d`. Author implementation; whole-sample review remains open.

The native application now connects pure stages and the SDK adapter to offline
source-indexed observations. CLI parsing, manifest/file handling and report IO are
outside inference. Files must have decimal source indices, no duplicates, exactly
1080 bytes of finite little-endian float32. When a manifest exists, its contract,
indices, paths and digests are checked. Each action dump contains 12 owned float32
values (48 bytes) and keeps its input source index.

Outputs and reports use exclusive creation; existing results are never replaced.
The report persists running/completed/failed state, partial records, input/output
hashes, model/manifest identity, SDK metadata, completed warmups and timing scope.
Aggregate latency/FPS belongs only to a complete run. Model/manifest changes fail
the run. Forced termination or write failure can leave incomplete evidence.

The Python launcher reuses the shared publication selection and board/hash checks
before CMake. Help/list/preview do not build, download or load SDK. The binary
independently repeats production preflight. Legacy model_path/input_path/output_dir
spellings remain aliases; unified options use hyphens. Native execution has no
implicit download. CMake separates host core tests, SDK and CLI targets; no fixture
implementation is part of the production target.

Verification: CLI and launcher tests failed before implementation and then passed.
Real native application compiled with an explicit runner/preflight double covers
21-input output/report generation, digest mismatches, parameter errors, existing
output preservation and an injected mid-run failure (one completed output after
two warmups). Launcher checks help/list/preview and refusal before build. Complete
HIMLoco suite: 23 tests pass. CMake Release host core build and CTest pass with
assertions retained in the test target. Shell syntax and migration contract pass.

- [Host test/gate commands](evidence/2026-09-28-b10-himloco-native-cli/host-checks.json)
- [CMake host build](evidence/2026-09-28-b10-himloco-native-cli/cmake-host.json)
- [Sample README contract](evidence/2026-09-28-b10-himloco-native-cli/contract.json)
- [Migration contract](evidence/2026-09-28-b10-himloco-native-cli/migration-contract.json)

Root and C++ bilingual README now describe actual commands, prerequisites,
parameter defaults, build switches, legacy aliases, public stages, resource
ownership, output provenance and failure interpretation. Historical measurements
remain attributed to the source. No actual model inference, real SDK compilation,
board test, download or quantization recipe execution occurred. User excludes
quantization reruns from migration completion requirements. Whole-sample quality
review, remaining consolidation and broader H0–H9 work stay open.
