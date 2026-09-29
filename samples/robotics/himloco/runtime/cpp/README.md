# HIMLoco C++ runtime

[中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards and verification

The unified entry includes the four-stage policy, SDK adapter, offline CLI and
build launcher. It retains float32 input/output semantics from X5 commit
`ac115717197920355fc390bb04299b20e6436864`, without extra normalization,
action scaling or robot control.

| Target | Artifact | Status |
| --- | --- | --- |
| X5 | Bayes-e fused Go2 BIN | Core, SDK-double and CLI host checks pass; real SDK compilation/board tests not-run |
| S100 / S100P / S600 | No matching publication | Not supported |

Host doubles do not establish hardware inference. The source environment was
RDK OS 3.5.0-beta, DNN Runtime 1.24.5 and HBRT 3.15.55; historical measurements
remain in the [evaluator guide](../../evaluator/README.md).

<a id="dependencies"></a>
## Dependencies

- Host core: C++17, CMake ≥ 3.18; direct compilation with `c++` also works.
- Native executable: X5 BSP `dnn/hb_dnn.h`, `dnn/hb_sys.h`, libdnn and
  `nlohmann/json.hpp` (usually from `nlohmann-json3-dev`). No gflags or OpenCV.
- Python launcher: Python, NumPy, PyYAML for unified selection and board checks;
  no Python inference SDK is loaded. Set `PYTHON` to choose its interpreter.

Published model inference requires neither Torch nor a quantization toolchain.
Fixture headers are only for host tests and must not replace the vendor SDK.

<a id="build"></a>
## Build

Run all commands from repository root. Without a board, build the core and host check:

```bash
cmake -S samples/robotics/himloco/runtime/cpp -B /tmp/himloco-host \
  -DHIMLOCO_BUILD_TESTS=ON
cmake --build /tmp/himloco-host --parallel 2
ctest --test-dir /tmp/himloco-host --output-on-failure
```

The launcher builds the native executable automatically on X5. For a manual build
(on X5 or in a configured cross-compilation environment):

```bash
cmake -S samples/robotics/himloco/runtime/cpp -B /tmp/himloco-native \
  -DHIMLOCO_BUILD_SDK=ON -DHIMLOCO_BUILD_CLI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/himloco-native --parallel 2
```

Output: `/tmp/himloco-native/himloco_cpp`. All three `HIMLOCO_BUILD_*` options
default to OFF; CLI requires SDK=ON. Set `HIMLOCO_DNN_INCLUDE`,
`HIMLOCO_DNN_LIBRARY` and `HIMLOCO_JSON_INCLUDE` for nonstandard dependencies;
use a CMake toolchain file for cross-compilation. CMake does not infer board type.
Execution independently checks actual board identity and model digest.

<a id="run"></a>
## Run

Preview on any host without downloads, build directories or SDK loading:

```bash
bash samples/robotics/himloco/runtime/cpp/run.sh --help
bash samples/robotics/himloco/runtime/cpp/run.sh --list-models
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 --dry-run
```

Prepare the publication explicitly, then execute on X5:

```bash
bash samples/robotics/himloco/model/download_model.sh --target x5
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 \
  --output-dir outputs/himloco_cpp
```

The default warms up 10 times, processes 21 observations, and writes 21 action
BINs plus `report.json`. Use a new output directory each time; the launcher never
downloads implicitly. A single-file run with explicit scheduling:

```bash
bash samples/robotics/himloco/runtime/cpp/run.sh --target x5 \
  --input-path samples/robotics/himloco/test_data/obs_history/000003.bin \
  --output-dir outputs/himloco_cpp_single --warmup 0 --priority 7
```

After a manual build, `/tmp/himloco-native/himloco_cpp` can run directly from
repository root using the defaults below. An explicit `--model-path` also requires
the exact `--asset-id`, and the file must still match the published digest.

<a id="parameters"></a>
## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--target` | Launcher `auto`; native `x5` | X5 only for execution; host dry-run requires explicit x5 |
| `--asset-id` | Launcher resolves; native fixed publication | `x5:himloco:himloco_go2_bayese_1x270.bin`; explicit for external paths |
| `--model-path` | `samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin` | Launcher default is absolute; binary default is relative to cwd |
| `--input-path` | `samples/robotics/himloco/test_data/obs_history` | Numeric BIN or directory; launcher default is absolute |
| `--output-dir` | `outputs/himloco_cpp` | Relative to cwd; must be new |
| `--report` | Output directory/`report.json` | Must be new; external location allowed |
| `--warmup` | `10` | First-input warmup, 0..1000000 |
| `--priority` | `-1` | SDK default, or 0..255 |
| `--build-dir` | `build` under this directory | Launcher only, native build location |
| `--dry-run` | `false` | Launcher only, show selection and build/run argv |
| `--list-models` | `false` | Launcher only, list the single publication |
| `--help` | `false` | Show usage |

Legacy aliases `--model_path`, `--input_path`, `--output_dir` are retained.
The binary accepts `--key value` and `--key=value`, rejecting duplicates and
unknown options. Custom relative paths passed to the launcher use the caller's cwd.

Input must contain 1080 bytes of finite little-endian float32. Numeric filename
stems identify source rows; duplicate indices are rejected. When
`runtime-input-manifest.json` exists above the input directory, its physical
contract, indices, paths and per-file digests are checked. Custom input without
a manifest is allowed and reports null manifest provenance. See the
[input guide](../../test_data/README.md).

<a id="interface-lifecycle"></a>
## Interface and resource lifecycle

`HimLoco(Runner)` stores the supplied callable and does not open a model or allocate
SDK buffers. Its public inference surface is limited to four stages:

```cpp
auto prepared = task.pre_process(observation);
auto raw = task.forward(prepared);
auto result = task.post_process(raw);
// Equivalent complete path:
auto complete = task.predict(observation);
```

`pre_process` validates and copies observations. `forward` validates its input,
invokes the runner once, then validates its returned actions and timing.
`post_process` returns an owned copy without clipping, normalization or the
controller's 0.25 rad action scaling. `predict` composes the three stages.

Invalid sizes, NaN/Inf values, invalid timing or an empty runner raise
`std::invalid_argument`. Runner failures propagate to the application. Returning
values instead of modifying caller-owned output parameters prevents a failed
prediction from presenting an earlier result as a new successful one.

All stage containers own their vectors. A later call cannot overwrite an earlier
raw output or its duration; preprocessing and postprocessing do not call the
runner. The task stores no mutable per-call state. Thread safety still depends
on the supplied runner; do not invoke a shared native SDK runner concurrently.
The adapter/application owns SDK loading, metadata, resource lifetime, input/output
files, reports and target/model identity checks; SDK lifecycle, metadata and identity checks are implemented in `SdkRunner`;
`cli_io.cc` and `application.cc` own input/output and reports.

### SDK adapter

`SdkRunner(NativeConfig)` first reads actual board identity, requires X5 and checks
the explicit local `.bin` path against the published SHA-256, before loading the
SDK. There is no download, environment override or S-platform fallback.
`NativeConfig::priority` defaults to `-1` (SDK default), with `[0,255]` accepted;
`model_path` must be supplied explicitly.

```cpp
himloco::SdkRunner runner({model_path, 7});
himloco::HimLoco task([&runner](const std::vector<float>& input) {
  return runner.run(input);
});
auto result = task.predict(observation);
```

The runner must outlive any task referencing it; do not use the same SDK resources
concurrently. The adapter requires one model, one `obs_history` input and one
`actions` output, float32 without manual dequantization. It checks four-dimensional
X5 logical/aligned shapes, element counts and allocation capacity before copying.
Input retains the source compact-submission convention, with remaining memory
zeroed. Output extraction follows aligned strides to return 12 owned logical values.
Task handles release on every exit; failed construction and destruction release
tensors before the packed model. `input_metadata()`/`output_metadata()`,
`model_name()`, `runtime_version()` and `priority()` supply report facts to the
application without writing reports inside inference code.

Host doubles exercise capacity/alignment, dtype/quantization rejection, allocation,
infer/wait/cache failures, nonfinite output and cleanup. Production preflight
rejections are checked separately with a board-identity reader double.
Real SDK compilation/execution has not run; the CLI has host end-to-end checks with a separately linked runner double.

<a id="results-interpretation"></a>
## Interpreting results

The 12 actions are unscaled policy outputs, not robot commands. The source
controller applies `default_joint_position + 0.25 * actions` outside the model
boundary. No live control loop is included.

The core reports the duration supplied with each `RawOutputs`; it does not add
preprocessing or postprocessing time. The native runner measures `hbDNNInfer` plus `hbDNNWaitTaskDone`, excluding
cache maintenance, input copying, output extraction and file I/O. The [evaluator guide](../../evaluator/README.md) retains
historical source measurements and their scopes; none is a new unified C++ result.
Offline numerical agreement alone does not establish closed-loop behavior.

Code follows the repository [Apache-2.0 license](../../../../../LICENSE).

### Output files and failures

Action filenames retain source indices, e.g. `000003.bin`, with 48 bytes each
(12 little-endian float32 values). Reports include model/manifest digests,
per-record input/output digests, SDK metadata, completed warmups, scheduling and
timings. `status: completed` means all records finished; `failed` retains the
error, current source index and completed records without aggregate latency.
Successful summaries include min/mean/p50/p95/max and sequential FPS when mean>0.
Exit 0 means completion/help; 2 means execution/argument error. Launcher build
failure also returns 2.

Files are exclusively created without replacing prior results. A changed model
or manifest fails the run. Forced termination or filesystem write failure can
leave a running or incomplete report; neither establishes success.
