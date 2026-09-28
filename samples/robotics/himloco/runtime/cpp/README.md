# HIMLoco C++ policy core

[中文](README_cn.md)

<a id="supported-boards"></a>
## Supported boards and migration status

The SDK-independent policy stages are implemented here. The native SDK adapter is implemented and checked with explicit host doubles;
the executable and launcher are still being migrated; this directory does
not yet provide a board inference command. The original implementation remains
in the [source C++ guide](../../../../../platforms/x5/samples/robotics/himloco/runtime/cpp/README.md).
Use the unified [Python entry](../python/README.md) for the implemented runtime
interface. Do not interpret a host policy test as an SDK or board test.

| Target | Published artifact | Native status |
| --- | --- | --- |
| X5 | Bayes-e fused Go2 BIN | Pure stages host-tested; SDK fixture checks pass; CLI pending; board not-run |
| S100 / S100P / S600 | None for this policy | Not supported |

Source: X5 commit `ac115717197920355fc390bb04299b20e6436864`.
The source stage semantics are retained: exactly 270 finite float observations
and 12 finite float actions, without normalization, activation or action scaling.

<a id="dependencies"></a>
## Dependencies

The core uses C++17 and the standard library only. No libdnn, gflags, OpenCV,
Python, Torch, model download or conversion toolchain is needed to compile it.
Host checks below were run with the installed Apple Clang compiler; they do not
establish compatibility with a board SDK. The historical source board environment
was RDK OS 3.5.0-beta, DNN Runtime 1.24.5 and HBRT 3.15.55.

The actual `sdk_runner.cc` also requires X5 BSP headers `dnn/hb_dnn.h`,
`dnn/hb_sys.h` and libdnn. `model_preflight.cc` reuses board identity and SHA-256
from `samples/_shared/cpp/`. Test fixture headers must never replace production SDK headers.

<a id="build"></a>
## Build the host contract check

Run from repository root:

```bash
build_dir="$(mktemp -d)"
c++ -std=c++17 -Wall -Wextra -Werror \
  -I samples/robotics/himloco/runtime/cpp \
  samples/robotics/himloco/runtime/cpp/policy.cc \
  samples/robotics/himloco/runtime/cpp/tests/test_policy.cc \
  -o "$build_dir/test_policy"
```

This builds an SDK-free test executable with an explicitly injected numerical
runner. It does not build or simulate a trained policy or quantized model.

<a id="run"></a>
## Run the host contract check

In the same shell, still at repository root:

```bash
"$build_dir/test_policy" samples/robotics/himloco/test_data/obs_history
```

Expect four `passed` lines and exit 0. The last check reads the 21 bundled source
observations and verifies that preprocessing preserves every float byte.
The other checks cover owned stage values, per-call timing, rejected inputs and
outputs, and propagation of an injected transport error. No SDK is loaded.

<a id="parameters"></a>
## Parameters and data types

There are no unified native CLI flags yet. The host test takes exactly one
positional argument: the bundled `obs_history` directory. The library interface
uses these types from [policy.hpp](policy.hpp):

| Type / field | Contract |
| --- | --- |
| `PreparedInput::values` | Owned vector of 270 finite float32 values |
| `RawOutputs::actions` | Owned vector of 12 finite float32 values |
| `RawOutputs::latency_ms` | Finite, nonnegative duration for this runner call |
| `InferenceResult` | Owned unchanged actions and the corresponding duration |
| `Runner` | `std::function<RawOutputs(const std::vector<float>&)>`; required, nonempty |

Observations contain the current 45-value frame followed by five previous frames.
The caller must supply the training-equivalent history, clipping, scaling and
joint order. The task does not construct or update this history. See the
[input guide](../../test_data/README.md) for layout and source provenance.

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
file/report/CLI migration remains pending.

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
Real SDK compilation/execution has not run; CLI work remains open.

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
