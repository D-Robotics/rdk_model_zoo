# Host validation runner (maintainer tooling)

`run.py` is the single maintainer command that executes the complete host
gate for the unified source in one deterministic, isolated run. It is
**not** a user-facing inference CLI and adds no model capability: People and
Agents keep running the native sample entries (ADR-0003). It orchestrates
tests only — no downloads, no board SDK, no VLA submodule initialization,
no board execution.

```bash
python tools/host_validation/run.py --repo PATH --report PATH [--python PYTHON]
```

`--repo` is the checkout to validate (the runner may live in another
checkout). `--report` is the machine-readable JSON report path and **must
live outside the source tree** (a report inside the checkout is rejected:
it would pollute the worktree and defeat the source-identity gate). Build
artifacts and per-suite logs land in `host-validation-artifacts/` next to
it. `--python` selects the interpreter for the suites (default: the
interpreter running `run.py`). Exit code 0 means every section passed.

## What one run executes

| Section | Content |
| --- | --- |
| Python suites | every first-party `unittest` directory: all sample `tests/` dirs under `samples/` (all 51 native samples, including nested `samples/vision/yoloe/conversion/tests` and `evaluator/tests`), `samples/_shared/tests` (including `test_vla_integration.py` — a parent-repo gitlink/pin integrity check that runs without submodule init and never executes upstream ACT/Pi0 code), `tools/board_validation/tests`, `tools/sample_contract/tests`, `skills/tests`, and this directory's own tests |
| Sample coverage | discovery cross-checked against the accepted 51-row sample inventory (`docs/releases/unified-migration/2026-10-05-all-sample-coverage.json`) — **required proof**: a sample or sample `tests/` directory listed in the inventory deleted from the tree, an extra sample beside the inventory, or a checkout whose inventory is absent, unparseable, empty or carries malformed/duplicate rows fails the run with explicit structured reasons (rows are validated and reported, never filtered away; the enforced scope follows the reviewed inventory, and no sample count is hardcoded in the runner) |
| Native prerequisites | C++17 compiler, cmake/ctest, git, the portable resolver in `native_dependencies.py` (nlohmann-json, gflags, iconv) — declared mandatory; a missing prerequisite fails the run instead of hiding native coverage. libsndfile/libsamplerate are probed and reported (they gate the default-on ASR CTest flags) |
| Static contract | `tools/sample_contract/check.py --scope migration --parser-mode import` with its report preserved |
| Native CTest | the six host-safe projects with the full-gate flags **on by default**: `gemma4-e2b/tests/native` (C++17 + OpenCV + pinned platforms commit), `yoloe/runtime/cpp/tests` (`YOLOE_TEST_OPENCV=ON`: real OpenCV image/mask/pipeline tests plus SDK-double and CLI fixtures), `asr/runtime/cpp/tests` (`ASR_AUDIO_TESTS=ON` + `ASR_CLI_TESTS=ON`: real libsndfile/libsamplerate frontend and host CLI tests), `ultralytics_yolo/runtime/cpp/test` (shared helpers/descriptor adapters against narrow API doubles), `paraformer/runtime/cpp` (`PARAFORMER_BUILD_TESTS=ON` + `PARAFORMER_BUILD_IO=ON` + `PARAFORMER_SANITIZERS=ON`: sanitised contract/pipeline tests, the SDK-double test against explicit fake headers, preflight and prepared-feature checks over Git-tracked evidence files (relative paths, no download/export) and the `PARAFORMER_HOST_FIXTURE` CLI help test), and `himloco/runtime/cpp` (`HIMLOCO_BUILD_TESTS=ON`: the SDK-free numerical policy test over the checked-in `obs_history` fixture) — SDK doubles and host libraries only; no vendor SDK, no model files, no downloads, no production SDK project, no board SDK and no real inference. The two `runtime/cpp` projects keep their vendor SDK adapter and production CLI switches (`PARAFORMER_BUILD_SDK`/`PARAFORMER_BUILD_CLI`, `HIMLOCO_BUILD_SDK`/`HIMLOCO_BUILD_CLI`) mandated **OFF**: a configured source directory under `runtime/cpp` alone does not imply vendor execution — those OFF defaults are mandatory safety, not a reduction of host test scope, and an enabling `--cmake-define` is rejected before any build (typed CMake cache spellings such as `VAR:BOOL`/`VAR:STRING`/`VAR:PATH` are rejected outright at option parsing: CMake lets a later typed define override the bare mandated value, so a typed key is exactly how an OFF switch could otherwise be re-enabled or a default-ON flag quietly disabled without a recorded scope reduction; whitespace-padded false tokens are rejected too — CMake preserves the leading whitespace of a `-D` value, so `" OFF "` would leave the switch enabled, and only exact, unpadded CMake false constants (OFF/FALSE/0/NO/N/IGNORE and the empty value case-insensitively, the exact `NOTFOUND` token and any `*-NOTFOUND` value case-sensitively) are accepted no-op restatements) |
| Catalog | `npm run check` in `tools/catalog-publisher` (sources validation, Vitest suite, build, `catalog:check`) — run only under the Node `engines` range the package itself declares (an undeclared, unparseable or unsatisfied range fails with the reason) |

Each suite runs in its own subprocess, so identical test module names across
samples stay isolated. Results come from a machine-readable `unittest`
result: exact counts, per-skip identities/reasons, per-failure messages.

The run also verifies the historical Git objects the tree declares
(`samples/_shared/legacy_platforms.py`, the Gemma native CMake pin, the
catalog commit source): a missing pin is an explicit failure naming the
`git fetch` command — never a silent skip of the suites that read pinned
sources. Full clone history is therefore a prerequisite.

## Source identity (content-based, not HEAD-only)

Before and after the run the runner snapshots: HEAD, branch, the
`git status --porcelain` dirty-file list, and a deterministic sha256 digest
over **the content** of every tracked file (mode-160000 gitlinks excluded —
no upstream code exists to hash) plus every untracked file Git's ignore
rules do not exclude (so `__pycache__`, `node_modules` and other ignored
artifacts never count as drift). Any HEAD change, dirty-list change or
content-digest change during the run fails the gate, and the report records
both digests and the dirty provenance of a dirty-but-stable checkout. A
checkout that is not a Git repository (no resolvable HEAD) can never pass:
source identity is a prerequisite, not an optional annotation.

## Skip policy (strict by default)

| Category | Meaning | Strict mode |
| --- | --- | --- |
| `optional_export` | the declared optional export scope, exactly: Torch/FunASR/Ultralytics import failures in `*/conversion/tests` directories (recorded as `optional_missing`), and the Paraformer export-stage suite's conditional framework skip (`samples/speech/paraformer/tests/test_export_stages`) | allowed; always recorded with identity and reason, never counted as executed tests |
| `native_prerequisite` | missing C++ compiler, nlohmann-json, gflags, iconv or CMake | **rejected** (fails the run); `--allow-native-skips` records them instead |
| `conditional` | legitimately conditional (e.g. a board SDK being installed makes an import-failure path unreachable) | allowed, recorded |
| anything else | undeclared skip — including framework names (torch/funasr/ultralytics) in unrelated runtime, model or native suites: the optional scope never hides a dependency regression | **rejected** |

Zero discovered tests, missing declared directories, unexplained test
directories, worker crashes, loader crashes, per-suite timeouts, CTest
per-stage timeouts/executable failures, corrupt CTest discovery output,
declared-vs-run CTest count mismatches, an absent, malformed or mismatched
accepted sample inventory and source drift can never report success. CTest
case counts are recorded in their own section (actual cases of the projects
that ran — no hardcoded totals) and are deliberately not added to the
Python unittest totals.

## Options

| Option | Effect |
| --- | --- |
| `--python PYTHON` | interpreter used for the suites |
| `--timeout N` | per-suite timeout in seconds (default 1800) |
| `--suite SUBSTRING` | run only matching suites (repeatable; sets `ci_equivalent: false`) |
| `--list` | print discovery (suites, native-only dirs, sample coverage, pins, CTest registry + defaults) and exit |
| `--skip-ctest` / `--skip-contract` / `--skip-catalog` | skip one section, recorded explicitly (`skipped-explicit`); not CI-equivalent |
| `--allow-native-skips` | record native-prerequisite skips instead of rejecting them; not CI-equivalent |
| `--cmake` / `--ctest` | executable overrides (also found next to the interpreter, e.g. a pip-installed cmake) |
| `--cmake-define PROJECT:VAR=VALUE` | override a define for one CTest project; the full-gate defaults (`YOLOE_TEST_OPENCV`, `ASR_AUDIO_TESTS`, `ASR_CLI_TESTS`, `PARAFORMER_BUILD_TESTS`/`BUILD_IO`/`SANITIZERS`, `HIMLOCO_BUILD_TESTS` = `ON`) are merged in — turning a default-ON flag off via any CMake false constant is recorded as a scope reduction naming the raw value and sets `ci_equivalent: false`. Classification matches real CMake's own reading of a `-D` value (verified against CMake 4.4.4): the named constants OFF/FALSE/0/NO/N/IGNORE and the empty value compare case-insensitively; the exact `NOTFOUND` token and any value ending in `-NOTFOUND` compare case-sensitively (`notfound`/`X-notfound` are truthy, not false constants); trailing whitespace is stripped by CMake's `-D` caching itself (so `NO ` is a real OFF), while leading whitespace survives (so ` NO` and `" NO "` keep the flag on and record no reduction). The vendor SDK / production CLI switches (`PARAFORMER_BUILD_SDK`/`BUILD_CLI`, `HIMLOCO_BUILD_SDK`/`BUILD_CLI`) are mandated OFF: an override that enables one is rejected before any build, with the reason on stderr (only an exact, unpadded false-constant restatement is a no-op — CMake keeps leading whitespace on `-D` values, and the guard never guesses which padding CMake might strip, so whitespace-padded false-looking tokens are rejected fail-closed). Only the untyped `PROJECT:VAR=VALUE` spelling is accepted — typed CMake cache keys (`VAR:BOOL`, `VAR:STRING`, `VAR:PATH`, …) are unsupported syntax and rejected with an actionable error at option parsing, before any configure/build |

Any option that reduces scope (skipped sections, suite filters, allowed
native skips, disabled default flags) is recorded in the report with
`ci_equivalent: false`, so a bounded local run can never be mistaken for
the full CI gate.

## OpenCV resolution

Projects that `find_package(OpenCV)` are configured after a probe resolves
the package: plain `find_package` first (apt `libopencv-dev` resolves here),
then the documented `OPENCV_DIR`/`OpenCV_DIR` environment override, then
standard package-manager prefixes (`/opt/homebrew/opt/opencv*/lib/cmake/
opencv{4,5}`, `/usr/local/opt/opencv*/…`, `/usr/lib/<arch>-linux-gnu/cmake/
opencv4`) because Homebrew installs the config as `opencv5`, which CMake
does not find via the `opencv4` layout. The winning mode and directory are
recorded in the CTest section; a failure names the install commands. No
personal paths are hardcoded.

## Requirements

- Python 3.10 or 3.12 with `pip install -r tools/host_validation/requirements.txt`
  (numpy, opencv-python-headless, PyYAML, scipy, onnx, onnxruntime,
  pycocotools, pillow, ftfy, regex — plus the sample-declared core host
  dependencies `lap==0.5.12` and `cython-bbox==0.1.5`, the ByteTrack
  tracker's real CPU dependencies, and `jsonschema`, which the skills
  evidence validators require and refuse to auto-install; all three are
  mandatory, nothing auto-skips around them). SciPy is environment-marked:
  Darwin on Python ≥ 3.12 requires ≥ 1.17.1 (the 1.15.3 macOS arm64 wheel
  is unusable — `scipy.sparse.linalg` fails to import, scipy/scipy#25635),
  while Python 3.10 keeps the general ≥ 1.10 bound (SciPy 1.17 does not
  support 3.10; the failure is Darwin-specific). The opencv/pycocotools
  floors are the samples' own declared constraints (≥ 4.8 / ≥ 2.0.7): the
  gate failures once seen on opencv 4.12.0.88 and pycocotools 2.0.10 were
  defects in the Zoo's own fixtures/evaluator (a whole-`sys.modules`
  rollback around the pinned legacy load, and a `loadRes` call on a
  no-info document the evaluator contract never required), both fixed in
  the samples with fresh-process regressions — upgrading either library
  was never the fix. Torch/FunASR/Ultralytics stay optional by design;
  their export suites report `optional_export` results.
- C++17 compiler, CMake ≥ 3.18 + CTest, OpenCV C++ dev headers (Gemma and
  YOLOE OpenCV-gated tests require them), gflags and nlohmann-json
  (standard system roots or pkg-config; env overrides documented in
  `native_dependencies.py`), **libsndfile and libsamplerate** for the
  default-on ASR audio/CLI tests (`brew install libsndfile libsamplerate`
  / `apt install libsndfile1-dev libsamplerate0-dev`), pkg-config, git with
  the full history including the declared pins.
- Node.js ≥ 22.12, < 23 with `npm ci` in `tools/catalog-publisher` for the
  catalog section (the runner itself never installs anything; a Node
  outside the declared `engines` range fails the section instead of
  claiming an unsupported pass).

## CI

`.github/workflows/host-validation.yml` runs this command on Ubuntu with
Python 3.10 and 3.12 and on macOS, on pushes to `develop`/`main` and on
pull requests, with the full clone history (no submodules), the declared
native dependencies installed, and the report written under
`$RUNNER_TEMP` (never inside the workspace, so the content-identity gate
cannot be tripped by its own report) and uploaded even on failure. A green
local run is not a CI claim: CI results are only what the GitHub jobs show.

## Files

- `run.py` — orchestrator and per-suite worker (the `--repo/--report/--python`
  interface above).
- `test_run.py` — fixture tests for the runner (discovery, isolation, skip
  policy, timeouts, pins, source drift, CTest stage failures and the
  safe-default/prohibited-override guards of the registered projects,
  catalog engines, required sample inventory, report structure) against
  synthetic repositories.
- `requirements.txt` — core host dependency set with environment markers
  and the tested-version record.
- `native_dependencies.py` / `test_native_dependencies.py` — portable native
  dependency discovery shared with the LLM host tests (see the module
  docstring for the resolution order and environment overrides).
