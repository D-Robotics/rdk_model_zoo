# HIMLoco Python policy stages

[中文](README_cn.md)

This directory currently provides the offline policy core in `policy.py`. The
unified SDK adapter and CLI are still being migrated; the example below injects
explicit synthetic outputs and does not execute the published model. The complete
[source runtime guide](../../../../../platforms/x5/samples/robotics/himloco/runtime/python/README.md)
is retained as the migration reference, not a claim that its entry exists here.

<a id="environment"></a>
## Environment

The core requires Python and NumPy only. It imports no board SDK, Torch, robot
middleware or conversion toolchain. Run the commands below from repository root.
Use the repository's host Python environment with NumPy installed. Actual X5
inference will require the BSP-provided `hbm_runtime` matching the board libraries;
the core does not install or substitute that dependency.

<a id="usage"></a>
## Usage

Construct `HimLocoTask(runner)`, where `runner` accepts the physical mapping
`{"obs_history": float32[1,270]}` and returns `{"actions": float32[1,12]}`.
The SDK adapter is responsible for target, asset identity and actual model metadata.
This core never downloads a model or opens a device. Use `predict(observation)`
for normal composition, or the three explicit methods shown below.

<a id="parameters"></a>
## Parameters

| API input | Contract |
| --- | --- |
| `runner` | Required callable; no default or implicit SDK construction |
| `observation` | Exactly 270 finite real numeric values, flattened in input order and converted to float32 |
| `pre_process(...).tensors` | Owned, contiguous float32 `obs_history` `[1,270]` |
| `forward(tensors)` | Exact named physical input; returns `RawOutputs` containing raw actions and this call's latency |
| `post_process(raw)` | Requires `RawOutputs`; no instance-level “last result” dependency |

Flat `[270]`, batch `[1,270]` and history `[6,45]` arrays preserve the source's
flattening convention. Complex, boolean, string/object, non-finite and float32
overflow inputs are rejected. Passing raw sensor data does not prepare the
training observation automatically.

<a id="results"></a>
## Results

`HimLocoResult.actions` is an independently owned float32 `[1,12]` array.
No clipping, activation, reordering, dequantization or scaling is applied.
`latency_ms` measures the synchronous injected runner call, excluding the task's
copies, preprocessing and postprocessing; it is not accelerator-only latency.
Each raw result carries its own time, so processing A after running B cannot
silently report B's time. SDK exceptions propagate; there is no fallback action.

The source deployment applies `default_joint_position + 0.25 * actions` outside
the model boundary. This core returns actions only and does not issue robot commands.

<a id="integration-example"></a>
## Executable integration example

```bash
python - <<'PYCODE'
import numpy as np
from samples.robotics.himloco.runtime.python.policy import HimLocoTask

def fixture_runner(tensors):
    assert tensors["obs_history"].shape == (1, 270)
    return {"actions": np.arange(12, dtype=np.float32).reshape(1, 12)}

task = HimLocoTask(fixture_runner)
observation = np.zeros((6, 45), dtype=np.float32)
prepared = task.pre_process(observation)
raw = task.forward(prepared.tensors)
explicit = task.post_process(raw)
result = task.predict(observation)
np.testing.assert_array_equal(explicit.actions, result.actions)
print(result.actions.shape, result.actions.tolist())
PYCODE
```

Expected output: shape `(1, 12)` and actions 0 through 11. These values come from
the fixture, not a learned policy. The core stores no history or per-call context;
callers provide all six observations. Serialize access if the injected SDK runner
uses shared buffers; owned results do not imply a thread-safe device runtime.

<a id="stage-io"></a>
## Stage semantics and source fidelity

Each 45-value observation contains velocity commands (3), angular velocity (3,
source scale 0.25), projected gravity (3), relative joint positions (12), relative
joint velocities (12, source scale 0.05), and previous actions (12). The current
observation comes first, followed by five earlier observations. This task does not
apply those scales again, accumulate history or choose a joint ordering.

`pre_process` packs and owns features; `forward` performs one raw model call;
`post_process` validates and owns raw actions; `predict` composes these methods.
Source preprocessing was compared against all 21 archived observation files with
manifest digest checks. Postprocessing uses synthetic action outputs for numerical
comparison. No board/model accuracy, control stability or robot motion was tested.
The original source's mutable last-latency field and potentially aliased outputs
are replaced by per-call records and independent result storage.

<a id="troubleshooting"></a>
## Troubleshooting

Wrong observation count requires reconstructing the training-policy history, not
padding or truncating blindly. Wrong output dtype/name/shape requires checking the
bound model interface; the core will not silently cast an incompatible model's
output. Latency attached to a hand-built `RawOutputs` must be finite and nonnegative.
Unified CLI, model preparation, native runtime and complete sample documentation
remain in progress; do not use this core-only status as whole-sample acceptance.
