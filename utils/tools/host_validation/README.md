English | [简体中文](README_cn.md)

# Host validation runner

`run.py` executes the repository's host validation sections and writes a JSON
report with suite results, source identity and run scope. Run it from any
directory with a checkout and report path:

```bash
python utils/tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]
```

`--repo` selects the checkout. `--report` must be outside that checkout;
build artifacts and per-suite logs are written to `host-validation-artifacts/`
beside the report. `--python` selects the suite interpreter and defaults to the
interpreter running `run.py`. Exit status 0 means all selected sections passed.

## Sections

| Section | Scope |
| --- | --- |
| Catalog | `npm run check` in `utils/tools/catalog-publisher`, run first so the generated `dist/` artifact exists during the run (the sample suites read the `docs/release` manifests directly). |
| Python | First-party `unittest` directories under all 54 inventory-listed samples, `utils/py_utils/tests`, `utils/tools/board_validation/tests`, `utils/tools/sample_contract/tests`, `skills/tests`, and this runner's tests. |
| Sample inventory | Cross-checks discovered sample tests with the required `utils/tools/host_validation/sample-inventory.json`. |
| Static contract | Runs `utils/tools/sample_contract/check.py --scope maintained --parser-mode import` (strict rule set). |
| Native CTest | Runs the six host C++ projects listed below. |

Catalog runs first so the generated `dist/` artifact exists for any consumer
during the run. Install catalog dependencies in
`utils/tools/catalog-publisher` with `npm ci` before running the section.

## CI scope versus the full local gate

The `host-validation` workflow runs this runner on one default leg —
ubuntu-24.04 with Python 3.12 — with `--skip-catalog --skip-contract`:
those two sections are owned by the `Model catalog data` and
`sample-contract` workflows, which run on the same events, so no check runs
twice in one CI run (the skips are recorded in the report, whose
`full_gate: false` describes the delegated scope, not a lesser CI run).
Python 3.10 and the macOS leg are available as expanded compatibility
through `workflow_dispatch` (`compatibility: expanded`). Changes limited
to narrative Markdown skip the host gate through an explicit successful
classification step: Markdown read as prose anywhere in the tree —
including sample READMEs, whose CLI/link contract the `sample-contract`
workflow enforces on every PR. Markdown that programs consume (test
fixtures and `test_data`, skill packs, inputs under `utils/tools`, and the
`docs/release` catalog data) and every non-Markdown change still run the
full gate. The command above with no skip flags remains the
complete local gate: every section, strict rules, Python 3.10 or 3.12,
Linux or macOS.

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
The JSON report uses schema `host-validation-report/2`; each run records
section results, the selected scope, `full_gate`, and the source
identity: HEAD, branch and the `git status --porcelain` dirty-file list
before and after the run. `full_gate` is true only when the run covered
the complete local gate (no skipped section, suite filter, allowed native
skip or disabled default flag); delivery CI legitimately shows
`full_gate: false` because it delegates the catalog and contract sections
to their own workflows — the flag describes this run's scope, not whether
the run is CI. Any HEAD or dirty-list change during a run fails
the source-identity check; a checkout that is not a Git repository cannot
pass.

## Options

| Option | Effect |
| --- | --- |
| `--python PYTHON` | Select the interpreter for Python suites. |
| `--timeout N` | Set each suite timeout in seconds (default `1800`). |
| `--suite SUBSTRING` | Select matching suites; repeatable. The report sets `full_gate` to `false`. |
| `--list` | Print discovered suites, native-only directories, sample coverage, pins and CTest defaults. |
| `--skip-ctest`, `--skip-contract`, `--skip-catalog` | Skip a section and record it in the report. Delivery CI passes `--skip-catalog --skip-contract` because the `Model catalog data` and `sample-contract` workflows own those checks. |
| `--allow-native-skips` | Record missing native prerequisites as skips. |
| `--cmake PATH`, `--ctest PATH` | Select CMake and CTest executables. |
| `--cmake-define PROJECT:VAR=VALUE` | Set a CMake define for a CTest project; repeatable. Full-gate defaults are merged in. |

The full-gate defaults are `YOLOE_TEST_OPENCV=ON`, `ASR_AUDIO_TESTS=ON`,
`ASR_CLI_TESTS=ON`, `PARAFORMER_BUILD_TESTS=ON`, `PARAFORMER_BUILD_IO=ON`,
`PARAFORMER_SANITIZERS=ON` and `HIMLOCO_BUILD_TESTS=ON`. Setting one of these
flags to a CMake false value records a scope reduction and sets
`full_gate: false`. The four vendor SDK/CLI flags above remain `OFF`;
enabling them is rejected. The accepted override syntax is the untyped
`PROJECT:VAR=VALUE`; typed cache keys such as `VAR:BOOL`, `VAR:STRING` and
`VAR:PATH` are rejected. A false value used to restate an SDK/CLI `OFF` default
must be an exact, unpadded CMake false constant (`OFF`, `FALSE`, `0`, `NO`, `N`,
`IGNORE`, or empty case-insensitively; `NOTFOUND` and `*-NOTFOUND`
case-sensitively).

Skip categories are `optional_export` (framework imports in conversion tests
and the conditional Paraformer export-stage suite), `native_prerequisite`
(compiler, CMake, nlohmann-json, gflags or iconv), and `conditional`
(identity-scoped board conditions, e.g. the board-built HobotMot
`libhobot_mot_capi.so` being absent on a host — accepted only for that one
native-tracking module's exact skip reason). Other
skips are reported as unexpected. `--allow-native-skips` makes native
prerequisite skips reportable; optional and conditional skips include their
identity and reason.

## Requirements

- Python 3.10 or 3.12 with dependencies from
  `utils/tools/host_validation/requirements.txt`.
- C++17 compiler, CMake 3.18 or newer, CTest, OpenCV development files,
  nlohmann-json, gflags, iconv, pkg-config and Git with the full clone history
  for declared pins.
- libsndfile and libsamplerate for the default ASR audio and CLI tests.
- liblz4 for the LiteAvatar background-cache tests, loaded through the
  production `ctypes` library-discovery path: `liblz4-1` on Ubuntu;
  `brew install lz4` on macOS, where a python.org interpreter additionally
  needs `DYLD_LIBRARY_PATH="$(brew --prefix lz4)/lib"` because ctypes
  discovery does not search Homebrew. The workflow probes this exact path
  before the suites run.
- FFmpeg with `ffprobe` and the libx264 encoder for the LiteAvatar host
  fixture suites (remux/both-track verification and background
  preparation run real lavfi transcodes; the production runtime keeps
  using the board VPU codecs): the `ffmpeg` package on Ubuntu,
  `brew install ffmpeg` on macOS. The workflow probes both binaries and a
  tiny libx264 encode before the suites run, so a missing FFmpeg fails
  the gate instead of skipping those tests.
- Node.js 22.12 or newer and below 23, plus `npm ci` in
  `utils/tools/catalog-publisher` — only for the local full gate's catalog
  section (delivery CI runs the catalog check in its own workflow, which
  needs no Node in this runner).

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
