# Host validation runner

`run.py` executes the repository's host validation sections and writes a JSON
report with suite results, source identity and run scope. Run it from any
directory with a checkout and report path:

```bash
python tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]
```

`--repo` selects the checkout. `--report` must be outside that checkout;
build artifacts and per-suite logs are written to `host-validation-artifacts/`
beside the report. `--python` selects the suite interpreter and defaults to the
interpreter running `run.py`. Exit status 0 means all selected sections passed.

## Sections

| Section | Scope |
| --- | --- |
| Catalog | `npm run check` in `tools/catalog-publisher`; the build creates `dist/catalog.json` before Python suites read the catalog snapshot. |
| Python | First-party `unittest` directories under all 51 inventory-listed samples, `utils/py_utils/tests`, `tools/board_validation/tests`, `tools/sample_contract/tests`, `skills/tests`, and this runner's tests. |
| Sample inventory | Cross-checks discovered sample tests with `docs/releases/unified-migration/2026-10-05-all-sample-coverage.json`. |
| Static contract | Runs `tools/sample_contract/check.py --scope migration --parser-mode import`. |
| Native CTest | Runs the six host C++ projects listed below. |

Catalog runs first because its build creates the file read by the ultralytics_yolo
asset and manifest snapshot suites. Install catalog dependencies in
`tools/catalog-publisher` with `npm ci` before running the section.

| Project | Source directory | Full-gate CMake flags and coverage |
| --- | --- | --- |
| Gemma native | `samples/llm/gemma4-e2b/tests/native` | C++17, OpenCV, pinned platforms commit |
| YOLOE | `samples/vision/yoloe/runtime/cpp/tests` | `YOLOE_TEST_OPENCV=ON`; OpenCV image, mask and pipeline tests, SDK-double tests and CLI fixtures |
| ASR | `samples/speech/asr/runtime/cpp/tests` | `ASR_AUDIO_TESTS=ON`, `ASR_CLI_TESTS=ON`; libsndfile/libsamplerate frontend and host CLI tests |
| Ultralytics YOLO common | `samples/vision/ultralytics_yolo/runtime/cpp/test` | Shared helpers and descriptor adapters with API doubles |
| Paraformer | `samples/speech/paraformer/runtime/cpp` | `PARAFORMER_BUILD_TESTS=ON`, `PARAFORMER_BUILD_IO=ON`, `PARAFORMER_SANITIZERS=ON`; host contracts, pipeline, SDK-double, preflight, prepared-feature and CLI-help tests |
| HIMLoco | `samples/robotics/himloco/runtime/cpp` | `HIMLOCO_BUILD_TESTS=ON`; numerical policy tests with the checked-in `obs_history` fixture |

`PARAFORMER_BUILD_SDK`, `PARAFORMER_BUILD_CLI`, `HIMLOCO_BUILD_SDK` and
`HIMLOCO_BUILD_CLI` default to `OFF`. These flags select vendor SDK adapters and
production CLIs; the host CTest projects use host libraries and SDK doubles.

Python suite results include exact test counts, skip identities and reasons,
and failure messages. CTest case counts appear in a separate report section.
The JSON report uses schema `host-validation-report/1`; each run records section
results, the selected scope, `ci_equivalent`, source HEAD/branch/dirty files,
and before/after SHA-256 digests of tracked file contents and non-ignored
untracked files. Gitlinks are recorded as pins rather than hashed file content.
Source changes during a run fail the source-identity check.

## Options

| Option | Effect |
| --- | --- |
| `--python PYTHON` | Select the interpreter for Python suites. |
| `--timeout N` | Set each suite timeout in seconds (default `1800`). |
| `--suite SUBSTRING` | Select matching suites; repeatable. The report sets `ci_equivalent` to `false`. |
| `--list` | Print discovered suites, native-only directories, sample coverage, pins and CTest defaults. |
| `--skip-ctest`, `--skip-contract`, `--skip-catalog` | Skip a section and record it in the report. `--skip-catalog` omits the catalog build needed by ultralytics_yolo snapshot suites. |
| `--allow-native-skips` | Record missing native prerequisites as skips. |
| `--cmake PATH`, `--ctest PATH` | Select CMake and CTest executables. |
| `--cmake-define PROJECT:VAR=VALUE` | Set a CMake define for a CTest project; repeatable. Full-gate defaults are merged in. |

The full-gate defaults are `YOLOE_TEST_OPENCV=ON`, `ASR_AUDIO_TESTS=ON`,
`ASR_CLI_TESTS=ON`, `PARAFORMER_BUILD_TESTS=ON`, `PARAFORMER_BUILD_IO=ON`,
`PARAFORMER_SANITIZERS=ON` and `HIMLOCO_BUILD_TESTS=ON`. Setting one of these
flags to a CMake false value records a scope reduction and sets
`ci_equivalent: false`. The four vendor SDK/CLI flags above remain `OFF`;
enabling them is rejected. The accepted override syntax is the untyped
`PROJECT:VAR=VALUE`; typed cache keys such as `VAR:BOOL`, `VAR:STRING` and
`VAR:PATH` are rejected. A false value used to restate an SDK/CLI `OFF` default
must be an exact, unpadded CMake false constant (`OFF`, `FALSE`, `0`, `NO`, `N`,
`IGNORE`, or empty case-insensitively; `NOTFOUND` and `*-NOTFOUND`
case-sensitively).

Skip categories are `optional_export` (framework imports in conversion tests
and the conditional Paraformer export-stage suite), `native_prerequisite`
(compiler, CMake, nlohmann-json, gflags or iconv), and `conditional`. Other
skips are reported as unexpected. `--allow-native-skips` makes native
prerequisite skips reportable; optional and conditional skips include their
identity and reason.

## Requirements

- Python 3.10 or 3.12 with dependencies from
  `tools/host_validation/requirements.txt`.
- C++17 compiler, CMake 3.18 or newer, CTest, OpenCV development files,
  nlohmann-json, gflags, iconv, pkg-config and Git with the full clone history
  for declared pins.
- libsndfile and libsamplerate for the default ASR audio and CLI tests.
- Node.js 22.12 or newer and below 23, plus `npm ci` in
  `tools/catalog-publisher`.

OpenCV lookup uses `find_package(OpenCV)`, the `OPENCV_DIR`/`OpenCV_DIR`
environment override, then standard Homebrew and Linux CMake package paths.

## Reported native ASR interface

The C++ `asr_demo` executable has a separate interface. Required options have
no default; unknown, duplicate and missing-value options fail.

| Option | Default |
| --- | --- |
| `--target` | Required: `s100` or `s600` |
| `--asset-id` | Required: `s:asr:TARGET/asr.hbm` |
| `--model-path` | Required |
| `--model-sha256` | Required: 64 hexadecimal digits |
| `--audio-file` | `samples/speech/asr/test_data/chi_sound.wav` |
| `--vocab-file` | `samples/speech/asr/test_data/vocab.json` |
| `--output-dir` | `outputs/asr_cpp/result` |
| `--decode-mode` | `ctc` |
| `--help` | `false` |

## Files

- `run.py` — runner and per-suite worker.
- `test_run.py` — runner fixture tests.
- `requirements.txt` — host Python dependencies.
- `native_dependencies.py` — native dependency discovery shared by host tests.
